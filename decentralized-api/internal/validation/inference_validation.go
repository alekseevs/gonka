package validation

import (
	"bytes"
	"decentralized-api/apiconfig"
	"decentralized-api/broker"
	"decentralized-api/chainphase"
	"decentralized-api/completionapi"
	"decentralized-api/cosmosclient"
	"decentralized-api/internal/utils"
	"decentralized-api/logging"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"sort"
	"sync"
	"time"

	"github.com/cosmos/cosmos-sdk/types/query"
	"github.com/google/uuid"
	"github.com/productscience/inference/api/inference/inference"
	"github.com/productscience/inference/x/inference/calculations"
	"github.com/productscience/inference/x/inference/types"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type InferenceValidator struct {
	recorder      cosmosclient.CosmosMessageClient
	nodeBroker    *broker.Broker
	configManager *apiconfig.ConfigManager
	phaseTracker  *chainphase.ChainPhaseTracker
}

func NewInferenceValidator(
	nodeBroker *broker.Broker,
	configManager *apiconfig.ConfigManager,
	recorder cosmosclient.CosmosMessageClient,
	phaseTracker *chainphase.ChainPhaseTracker) *InferenceValidator {
	return &InferenceValidator{
		nodeBroker:    nodeBroker,
		configManager: configManager,
		recorder:      recorder,
		phaseTracker:  phaseTracker,
	}
}

func (s *InferenceValidator) VerifyInvalidation(events map[string][]string, recorder cosmosclient.InferenceCosmosClient) {
	inferenceIds, ok := events["inference_validation.inference_id"]
	if !ok || len(inferenceIds) == 0 {
		logging.Error("No inference_id found in events", types.Validation)
		return
	}
	inferenceId := inferenceIds[0]

	logging.Debug("Verifying invalidation", types.Validation, "inference_id", inferenceId)

	queryClient := recorder.NewInferenceQueryClient()

	r, err := queryClient.Inference(recorder.GetContext(), &types.QueryGetInferenceRequest{Index: inferenceId})
	if err != nil {
		// FIXME: what should we do with validating the transaction?
		logging.Warn("Failed to query Inference for revalidation.", types.Validation, "error", err)
		return
	}

	logInferencesToValidate([]string{inferenceId})
	go func() {
		s.validateInferenceAndSendValMessage(r.Inference, recorder, true)
	}()

}

// shouldValidateInference determines if the current participant should validate a specific inference
// This function extracts the core validation decision logic for reuse in recovery scenarios
func (s *InferenceValidator) shouldValidateInference(
	inferenceDetails *types.InferenceValidationDetails,
	seed int64,
	validatorPower uint64,
	validatorAddress string,
	validationParams *types.ValidationParams,
) (bool, string) {
	// Skip if this participant is the executor
	if inferenceDetails.ExecutorId == validatorAddress {
		return false, "Skipping validation: participant is the executor"
	}

	// Skip if total power is invalid
	if inferenceDetails.TotalPower <= inferenceDetails.ExecutorPower {
		return false, "Skipping validation: total power is less than or equal to executor power"
	}

	// Use the same validation logic as real-time validations
	shouldValidate, message := calculations.ShouldValidate(
		seed,
		inferenceDetails,
		uint32(inferenceDetails.TotalPower),
		uint32(validatorPower),
		uint32(inferenceDetails.ExecutorPower),
		validationParams)

	return shouldValidate, message
}

func (s *InferenceValidator) getNodeModelsAtEpoch(epochIndex uint64, address string) (map[string]bool, error) {
	supportedModels := make(map[string]bool)
	parentEpochData, err := s.nodeBroker.GetChainBridge().GetEpochGroupDataByModelId(epochIndex, "")
	if err != nil {
		logging.Error("Failed to get epoch group data by model id", types.ValidationRecovery, "error", err)
		return nil, fmt.Errorf("failed to get epoch group data by model id: %w", err)
	}
	for _, modelId := range parentEpochData.EpochGroupData.SubGroupModels {
		subgroupResp, err := s.nodeBroker.GetChainBridge().GetEpochGroupDataByModelId(parentEpochData.EpochGroupData.EpochIndex, modelId)
		if err != nil {
			logging.Error("Failed to get subgroup epoch data", types.ValidationRecovery, "model_id", modelId, "error", err)
			continue
		}
		if subgroupResp == nil {
			logging.Warn("Subgroup epoch data response is nil", types.ValidationRecovery, "model_id", modelId)
			continue
		}

		subgroup := subgroupResp.EpochGroupData
		if subgroup.ModelSnapshot == nil {
			logging.Error("ModelSnapshot is nil in subgroup", types.ValidationRecovery, "model_id", modelId)
			continue
		}

		for _, weightInfo := range subgroup.ValidationWeights {
			if weightInfo.MemberAddress == address {
				supportedModels[modelId] = true
			}
		}
	}
	logging.Info("Supported models at epoch", types.ValidationRecovery, "epochIndex", epochIndex, "supportedModels", supportedModels, "address", address)

	return supportedModels, nil
}

func (s *InferenceValidator) getCurrentSupportedModels() (map[string]bool, error) {
	supportedModels := make(map[string]bool)
	nodes, err := s.nodeBroker.GetNodes()
	if err != nil {
		logging.Error("Failed to get nodes from broker", types.ValidationRecovery, "error", err)
		return nil, fmt.Errorf("failed to get nodes: %w", err)
	}
	for _, node := range nodes {
		nodeState := node.State
		for model := range nodeState.EpochModels {
			supportedModels[model] = true
		}
	}
	logging.Debug("Supported models", types.ValidationRecovery, "supportedModels", supportedModels)
	return supportedModels, nil
}

// DetectMissedValidations identifies which validations were missed for a specific epoch
// Returns a list of inference objects that the current participant should have validated but didn't
func (s *InferenceValidator) DetectMissedValidations(epochIndex uint64, seed int64) ([]types.Inference, error) {
	logging.Info("Starting missed validation detection", types.ValidationRecovery, "epochIndex", epochIndex, "seed", seed)

	queryClient := s.recorder.NewInferenceQueryClient()
	address := s.recorder.GetAddress()

	// Get all inferences (automatically pruned to recent 2-3 epochs) with pagination
	var allInferences []types.Inference
	var nextKey []byte

	for {
		req := &types.QueryAllInferenceRequest{
			Pagination: &query.PageRequest{
				Key:   nextKey,
				Limit: 1000, // Use larger page size for efficiency
			},
		}

		resp, err := queryClient.InferenceAll(s.recorder.GetContext(), req)
		if err != nil {
			logging.Error("Failed to query inferences page", types.ValidationRecovery, "error", err)
			return nil, fmt.Errorf("failed to query inferences: %w", err)
		}

		allInferences = append(allInferences, resp.Inference...)

		// Check if there are more pages
		if resp.Pagination == nil || len(resp.Pagination.NextKey) == 0 {
			break
		}
		nextKey = resp.Pagination.NextKey
	}

	logging.Debug("Retrieved all inferences", types.ValidationRecovery, "totalCount", len(allInferences))

	// Filter inferences by epoch
	var epochInferences []types.Inference
	for _, inf := range allInferences {
		if inf.EpochId == epochIndex {
			epochInferences = append(epochInferences, inf)
		}
	}

	if len(epochInferences) == 0 {
		logging.Info("No inferences found for epoch", types.ValidationRecovery, "epochIndex", epochIndex)
		return []types.Inference{}, nil
	}

	logging.Info("Found inferences for epoch", types.ValidationRecovery, "epochIndex", epochIndex, "count", len(epochInferences))

	// Create a map for quick lookup of inferences by ID
	inferenceMap := make(map[string]types.Inference)
	inferenceIds := make([]string, len(epochInferences))
	for i, inf := range epochInferences {
		inferenceIds[i] = inf.InferenceId
		inferenceMap[inf.InferenceId] = inf
	}

	validationParamsResp, err := queryClient.GetInferenceValidationParameters(s.recorder.GetContext(), &types.QueryGetInferenceValidationParametersRequest{
		Ids:       inferenceIds,
		Requester: address,
	})
	if err != nil {
		logging.Error("Failed to get validation parameters", types.ValidationRecovery, "error", err)
		return nil, fmt.Errorf("failed to get validation parameters: %w", err)
	}

	// Get validation params
	params, err := queryClient.Params(s.recorder.GetContext(), &types.QueryParamsRequest{})
	if err != nil {
		logging.Error("Failed to get params", types.ValidationRecovery, "error", err)
		return nil, fmt.Errorf("failed to get params: %w", err)
	}

	// Get what validations were already submitted by this participant
	epochGroupValidationsResp, err := queryClient.EpochGroupValidations(s.recorder.GetContext(), &types.QueryGetEpochGroupValidationsRequest{
		Participant: address,
		EpochIndex:  epochIndex,
	})

	// Create a set of already validated inference IDs
	alreadyValidated := make(map[string]bool)
	if err == nil {
		for _, inferenceId := range epochGroupValidationsResp.EpochGroupValidations.ValidatedInferences {
			alreadyValidated[inferenceId] = true
		}
	} else {
		if status.Code(err) == codes.NotFound {
			logging.Info("No epoch group validations found", types.ValidationRecovery, "participant", address, "epochIndex", epochIndex)
		} else {
			logging.Warn("Failed to get epoch group validations", types.ValidationRecovery, "error", err, "participant", address, "epochIndex", epochIndex)
		}
	}
	supportedModels, err := s.getNodeModelsAtEpoch(epochIndex, address)
	if err != nil {
		logging.Error("Failed to get supported models at epoch", types.ValidationRecovery, "error", err)
		return nil, fmt.Errorf("failed to get supported models at epoch: %w", err)
	}

	// Check each inference to see if it should have been validated but wasn't
	var missedValidations []types.Inference
	for _, inferenceDetails := range validationParamsResp.Details {
		if !supportedModels[inferenceDetails.Model] {
			logging.Info("Skipping inference - model not supported by any node", types.ValidationRecovery, "inferenceId", inferenceDetails.InferenceId, "model", inferenceDetails.Model)
			continue
		}
		// Check if this participant should validate this inference
		shouldValidate, message := s.shouldValidateInference(
			inferenceDetails,
			seed,
			validationParamsResp.ValidatorPower,
			address,
			params.Params.ValidationParams)

		logging.Debug("Validation check result", types.ValidationRecovery,
			"inferenceId", inferenceDetails.InferenceId,
			"shouldValidate", shouldValidate,
			"message", message,
			"alreadyValidated", alreadyValidated[inferenceDetails.InferenceId])

		// If should validate but didn't, add to missed list
		if shouldValidate && !alreadyValidated[inferenceDetails.InferenceId] {
			if inference, exists := inferenceMap[inferenceDetails.InferenceId]; exists {
				missedValidations = append(missedValidations, inference)
				logging.Info("Found missed validation", types.ValidationRecovery, "inferenceId", inferenceDetails.InferenceId)
			} else {
				logging.Warn("Inference not found in map", types.ValidationRecovery, "inferenceId", inferenceDetails.InferenceId)
			}
		}
	}

	logging.Info("Missed validation detection complete", types.ValidationRecovery,
		"epochIndex", epochIndex,
		"totalInferences", len(epochInferences),
		"missedValidations", len(missedValidations))

	return missedValidations, nil
}

// ExecuteRecoveryValidations executes validation for a list of missed inferences
// This function uses the inference data already obtained and executes validations in parallel goroutines
// It waits for all validations to complete before returning
func (s *InferenceValidator) ExecuteRecoveryValidations(missedInferences []types.Inference) (int, error) {

	availableModels, err := s.getCurrentSupportedModels()
	if err != nil {
		logging.Error("Failed to get currently available models", types.ValidationRecovery, "error", err)
		return 0, fmt.Errorf("failed to get currently available models: %w", err)
	}

	missedInferencesToValidate := []types.Inference{}
	for _, inf := range missedInferences {
		if availableModels[inf.Model] {
			missedInferencesToValidate = append(missedInferencesToValidate, inf)
		} else {
			logging.Info("Can't recover validation for inference, model not available", types.ValidationRecovery, "inferenceId", inf.InferenceId, "model", inf.Model)
		}
	}

	if len(missedInferences) > len(missedInferencesToValidate) {
		logging.Warn("Some inferences can't be recovered, model not available", types.ValidationRecovery, "missedInferences", len(missedInferences), "missedInferencesToValidate", len(missedInferencesToValidate))
	}

	if len(missedInferencesToValidate) == 0 {
		logging.Info("No missed validations to execute", types.ValidationRecovery)
		return 0, nil
	}

	logging.Info("Starting recovery validation execution", types.ValidationRecovery, "missedValidations", len(missedInferencesToValidate))

	var wg sync.WaitGroup

	// Execute recovery validations in parallel goroutines with WaitGroup synchronization
	for _, inf := range missedInferencesToValidate {
		wg.Add(1)
		go func(inference types.Inference) {
			defer wg.Done()

			logging.Info("Executing recovery validation", types.ValidationRecovery, "inferenceId", inference.InferenceId)

			// Use existing validation infrastructure
			// The validateInferenceAndSendValMessage function handles all validation logic, node locking, and message sending
			// Cast the interface back to concrete type (safe since it's always *InferenceCosmosClient)
			concreteRecorder := s.recorder.(*cosmosclient.InferenceCosmosClient)
			s.validateInferenceAndSendValMessage(inference, *concreteRecorder, false)

			logging.Info("Recovery validation completed", types.ValidationRecovery, "inferenceId", inference.InferenceId)
		}(inf)
	}

	// Wait for all recovery validations to complete
	logging.Info("Waiting for all recovery validations to complete", types.ValidationRecovery, "count", len(missedInferences))
	wg.Wait()

	logging.Info("All recovery validations completed", types.ValidationRecovery, "count", len(missedInferences))
	return len(missedInferencesToValidate), nil
}

func (s *InferenceValidator) WaitForValidationsToBeRecorded() {
	const maxTimeoutBlocks = 60
	epochLength := s.phaseTracker.GetEpochParams().EpochLength
	timeoutBlocks := min(epochLength/10, maxTimeoutBlocks)

	time.Sleep(5 * time.Duration(timeoutBlocks) * time.Second)
}

func (s *InferenceValidator) SampleInferenceToValidate(ids []string, transactionRecorder cosmosclient.InferenceCosmosClient) {
	if ids == nil {
		logging.Debug("No inferences to validate", types.Validation)
		return
	}

	logging.Debug("Sampling inf transactions to validate", types.Validation)

	queryClient := transactionRecorder.NewInferenceQueryClient()

	r, err := queryClient.GetInferenceValidationParameters(transactionRecorder.GetContext(), &types.QueryGetInferenceValidationParametersRequest{
		Ids:       ids,
		Requester: transactionRecorder.GetAddress(),
	})
	if err != nil {
		// FIXME: what should we do with validating the transaction?
		logging.Warn("Failed to query GetInferenceValidationParameters.", types.Validation, "error", err)
		return
	}

	params, err := queryClient.Params(transactionRecorder.GetContext(), &types.QueryParamsRequest{})
	if err != nil {
		logging.Error("Failed to get params", types.Validation, "error", err)
		return
	}

	supportedModels, err := s.getCurrentSupportedModels()
	if err != nil {
		logging.Error("Failed to get currently available models", types.Validation, "error", err)
		return
	}

	logInferencesToSample(r.Details)

	address := transactionRecorder.GetAddress()
	currentSeed := s.configManager.GetCurrentSeed().Seed
	var toValidateIds []string

	for _, inferenceWithExecutor := range r.Details {
		if !supportedModels[inferenceWithExecutor.Model] {
			logging.Debug("Skipping inference by not supported model", types.Validation, "inferenceId", inferenceWithExecutor.InferenceId, "model", inferenceWithExecutor.Model)
			continue
		}
		// Use the extracted validation decision logic
		shouldValidate, message := s.shouldValidateInference(
			inferenceWithExecutor,
			currentSeed,
			r.ValidatorPower,
			address,
			params.Params.ValidationParams)

		logging.Info(message, types.Validation, "inferenceId", inferenceWithExecutor.InferenceId, "seed", currentSeed, "validator", address)

		if shouldValidate {
			toValidateIds = append(toValidateIds, inferenceWithExecutor.InferenceId)
		}
	}

	logInferencesToValidate(toValidateIds)
	for _, inf := range toValidateIds {
		go func() {
			response, err := queryClient.Inference(transactionRecorder.GetContext(), &types.QueryGetInferenceRequest{Index: inf})
			if err != nil {
				logging.Error("Failed to get inference by id", types.Validation, "id", response, "error", err)
				return
			}
			s.validateInferenceAndSendValMessage(response.Inference, transactionRecorder, false)
		}()
	}
}

func logInferencesToSample(inferences []*types.InferenceValidationDetails) {
	var ids []struct {
		InferenceId string
		ExecutorId  string
	}

	for _, inf := range inferences {
		ids = append(ids, struct {
			InferenceId string
			ExecutorId  string
		}{
			InferenceId: inf.InferenceId,
			ExecutorId:  inf.ExecutorId,
		})
	}

	logging.Info("Inferences to sample", types.Validation, "ids", ids)
}

func logInferencesToValidate(toValidate []string) {
	var ids []string
	for _, inf := range toValidate {
		ids = append(ids, inf)
	}
	logging.Info("Inferences to validate", types.Validation, "inferences", ids)
}

func (s *InferenceValidator) validateInferenceAndSendValMessage(inf types.Inference, transactionRecorder cosmosclient.InferenceCosmosClient, revalidation bool) {
	const maxRetries = 5
	const retryInterval = 4 * time.Minute

	var valResult ValidationResult
	var err error

	// Retry logic for LockNode operation
	for attempt := 1; attempt <= maxRetries; attempt++ {
		valResult, err = broker.LockNode(s.nodeBroker, inf.Model, func(node *broker.Node) (ValidationResult, error) {
			return s.validate(inf, node)
		})

		if err == nil {
			// Success, break out of retry loop
			break
		}

		// For all errors, check if we should retry
		if attempt < maxRetries {
			logging.Warn("Failed to validate inference, retrying", types.Validation,
				"id", inf.InferenceId,
				"attempt", attempt,
				"maxRetries", maxRetries,
				"error", err,
				"nextRetryIn", retryInterval)
			time.Sleep(retryInterval)
		} else {
			// Final attempt failed - check if it's ErrNoNodesAvailable for special handling
			if errors.Is(err, broker.ErrNoNodesAvailable) {
				logging.Warn("Failed to validate inference after all retry attempts. No nodes available, probably unsupported model.", types.Validation, "id", inf.InferenceId, "attempts", maxRetries, "error", err)
				return
			} else {
				logging.Error("Failed to validate inference after all retry attempts", types.Validation,
					"id", inf.InferenceId,
					"attempts", maxRetries,
					"error", err)
				return
			}
		}
	}

	msgValidation, err := ToMsgValidation(valResult)
	if err != nil {
		logging.Error("Failed to convert to MsgValidation.", types.Validation, "id", inf.InferenceId, "error", err)
		return
	}
	msgValidation.Revalidation = revalidation

	if err = transactionRecorder.ReportValidation(msgValidation); err != nil {
		logging.Error("Failed to report validation.", types.Validation, "id", inf.InferenceId, "error", err)
		return
	}

	logging.Info("Successfully validated inference", types.Validation, "id", inf.InferenceId)
}

func (s *InferenceValidator) validate(inference types.Inference, inferenceNode *broker.Node) (ValidationResult, error) {
	logging.Debug("Validating inference", types.Validation, "id", inference.InferenceId)

	if inference.Status == types.InferenceStatus_STARTED {
		logging.Error("Inference not finished", types.Validation, "status", inference.Status, "inference", inference)
		return nil, errors.New("Inference is not finished. id = " + inference.InferenceId)
	}

	var requestMap map[string]interface{}
	if err := json.Unmarshal([]byte(inference.PromptPayload), &requestMap); err != nil {
		return &InvalidInferenceResult{inference.InferenceId, "Failed to unmarshal inference.PromptPayload.", err}, nil
	}

	originalResponse, err := unmarshalResponse(&inference)
	if err != nil {
		return &InvalidInferenceResult{inference.InferenceId, "Failed to unmarshal inference.ResponsePayload.", err}, nil
	}

	enforcedTokens, err := originalResponse.GetEnforcedTokens()
	if err != nil {
		return &InvalidInferenceResult{inference.InferenceId, "Failed to get enforced string.", err}, nil
	}

	// From here on, errors are on the part of the validator, not the inference that was passed in
	requestMap["enforced_tokens"] = enforcedTokens
	requestMap["stream"] = false
	requestMap["skip_special_tokens"] = false
	delete(requestMap, "stream_options")

	requestBody, err := json.Marshal(requestMap)
	if err != nil {
		return nil, err
	}

	completionsUrl, err := url.JoinPath(inferenceNode.InferenceUrlWithVersion(s.configManager.GetCurrentNodeVersion()), "v1/chat/completions")
	if err != nil {
		logging.Error("Failed to join url", types.Validation, "url", inferenceNode.InferenceUrlWithVersion(s.configManager.GetCurrentNodeVersion()), "error", err)
		return nil, err
	}

	resp, err := http.Post(
		completionsUrl,
		"application/json",
		bytes.NewReader(requestBody),
	)
	if err != nil {
		return nil, err
	}

	respBodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	logging.Debug("responseValidation", types.Validation, "validation", string(respBodyBytes))
	responseValidation, err := completionapi.NewCompletionResponseFromBytes(respBodyBytes)
	if err != nil {
		logging.Error("Failed to unmarshal responseValidation", types.Validation, "id", inference.InferenceId, "error", err)
		return nil, err
	}

	originalLogits := originalResponse.ExtractLogits()
	validationLogits := responseValidation.ExtractLogits()
	baseResult := BaseValidationResult{
		InferenceId:   inference.InferenceId,
		ResponseBytes: respBodyBytes,
	}
	if len(originalLogits) == 0 || len(validationLogits) == 0 {
		logging.Error("No logits found in original or validation response", types.Validation, "id", inference.InferenceId, "originalLogits", originalLogits, "validationLogits", validationLogits)
		return nil, errors.New("no logits found in original or validation response")
	}

	return compareLogits(originalLogits, validationLogits, baseResult), nil
}

func unmarshalResponse(inference *types.Inference) (completionapi.CompletionResponse, error) {
	resp, err := completionapi.NewCompletionResponseFromLinesFromResponsePayload(inference.ResponsePayload)

	if err != nil {
		logging.Error("Failed to unmarshal inference.ResponsePayload.", types.Validation, "id", inference.InferenceId, "error", err)
	}

	switch resp.(type) {
	case *completionapi.StreamedCompletionResponse:
		logging.Info("Unmarshalled inference.ResponsePayload into StreamedResponse", types.Validation, "id", inference.InferenceId)
	case *completionapi.JsonCompletionResponse:
		logging.Info("Unmarshalled inference.ResponsePayload into JsonResponse", types.Validation, "id", inference.InferenceId)
	default:
		logging.Error("Failed to unmarshal inference.ResponsePayload into StreamedResponse or JsonResponse", types.Validation, "id", inference.InferenceId)
	}

	return resp, err
}

type ValidationResult interface {
	GetInferenceId() string

	GetValidationResponseBytes() []byte

	IsSuccessful() bool
}

type BaseValidationResult struct {
	InferenceId   string
	ResponseBytes []byte
}

func (r BaseValidationResult) GetInferenceId() string {
	return r.InferenceId
}

func (r BaseValidationResult) GetValidationResponseBytes() []byte {
	return r.ResponseBytes
}

type DifferentLengthValidationResult struct {
	BaseValidationResult
}

func (DifferentLengthValidationResult) IsSuccessful() bool {
	return false
}

type DifferentTokensValidationResult struct {
	BaseValidationResult
}

func (DifferentTokensValidationResult) IsSuccessful() bool {
	return false
}

type SimilarityValidationResult struct {
	BaseValidationResult
	Value float64
}

func (r SimilarityValidationResult) IsSuccessful() bool {
	return r.Value > 0.99
}

type InvalidInferenceResult struct {
	InferenceId string
	Reason      string
	Error       error
}

func (r InvalidInferenceResult) IsSuccessful() bool {
	return false
}

func (r InvalidInferenceResult) GetInferenceId() string {
	return r.InferenceId
}

func (r InvalidInferenceResult) GetValidationResponseBytes() []byte {
	return []byte{}
}

func compareLogits(
	originalLogits []completionapi.Logprob,
	validationLogits []completionapi.Logprob,
	baseComparisonResult BaseValidationResult,
) ValidationResult {
	if len(originalLogits) != len(validationLogits) {
		logging.Warn("Different length of logits", types.Validation, "originalLogits", originalLogits, "validationLogits", validationLogits, "lengthOriginal", len(originalLogits), "lengthValidation", len(validationLogits))
	}
	if len(validationLogits) < len(originalLogits) {
		logging.Warn("Validation logits are shorter than original logits", types.Validation, "originalLogits", originalLogits, "validationLogits", validationLogits, "lengthOriginal", len(originalLogits), "lengthValidation", len(validationLogits))
		return &DifferentLengthValidationResult{baseComparisonResult}
	}

	for i := range originalLogits {
		o := originalLogits[i]
		v := validationLogits[i]
		if o.Token != v.Token {
			logging.Error("Different tokens in logits", types.Validation, "originalLogits", originalLogits, "validationLogits", validationLogits)

			return &DifferentTokensValidationResult{baseComparisonResult}
		}
	}
	similarity := customSimilarity(originalLogits, validationLogits)

	return &SimilarityValidationResult{BaseValidationResult: baseComparisonResult, Value: similarity}
}

func customSimilarity(
	originalLogprobs []completionapi.Logprob,
	validationLogprobs []completionapi.Logprob,
) float64 {
	distance, err := customDistance(originalLogprobs, validationLogprobs)
	if err != nil {
		logging.Error("Error calculating custom distance", types.Validation, "error", err)
		return 0
	}
	similarity := 1 - distance
	if similarity < 0 {
		logging.Error("Similarity value is negative", types.Validation, "similarity", similarity)
		return 0
	}
	return similarity
}

func customDistance(
	originalLogprobs []completionapi.Logprob,
	validationLogprobs []completionapi.Logprob,
) (float64, error) {
	distance := 0.0
	for i := range originalLogprobs {
		o := originalLogprobs[i]
		v := validationLogprobs[i]
		posDistance, err := positionDistance(o.TopLogprobs, v.TopLogprobs)
		if err != nil {
			logging.Error("Error calculating position distance", types.Validation, "error", err)
			return math.Inf(1), err
		}
		distance += posDistance
	}
	totalLogprobs := max(100, len(originalLogprobs)) * len(originalLogprobs[0].TopLogprobs)

	return distance / float64(totalLogprobs), nil
}

func positionDistance(
	originalLogprobs []completionapi.TopLogprobs,
	validationLogprobs []completionapi.TopLogprobs,
) (float64, error) {
	if len(originalLogprobs) == 0 || len(validationLogprobs) == 0 {
		return 0.0, fmt.Errorf("empty logprobs provided")
	}
	distance := 0.0

	originalLogprobMap := make(map[string]float64)
	for _, o := range originalLogprobs {
		originalLogprobMap[o.Token] = o.Logprob
	}
	sortedLogprobs := make([]float64, 0, len(originalLogprobMap))
	for _, logprob := range originalLogprobMap {
		sortedLogprobs = append(sortedLogprobs, logprob)
	}

	sort.Float64s(sortedLogprobs)

	var minOriginalLogprob1, minOriginalLogprob2 float64
	if len(sortedLogprobs) >= 2 {
		minOriginalLogprob1 = sortedLogprobs[0]
		minOriginalLogprob2 = sortedLogprobs[1]
	} else if len(sortedLogprobs) == 1 {
		minOriginalLogprob1 = sortedLogprobs[0]
		minOriginalLogprob2 = minOriginalLogprob1 - 100.0
	}

	// Estimate the next logprob value (2 as fine)
	nextOriginalLogprob := minOriginalLogprob1 - (minOriginalLogprob2 - minOriginalLogprob1)

	for _, v := range validationLogprobs {
		var originalLogprob float64
		if origProb, exists := originalLogprobMap[v.Token]; exists {
			originalLogprob = origProb
		} else {
			originalLogprob = nextOriginalLogprob
		}

		denom := 1e-6 + math.Abs(v.Logprob) + math.Abs(originalLogprob)
		distance += math.Abs(v.Logprob-originalLogprob) / denom / 2.0
	}

	return distance, nil
}

func ToMsgValidation(result ValidationResult) (*inference.MsgValidation, error) {
	// Match type of result from implementations of ValidationResult
	var simVal float64
	switch result.(type) {
	case *DifferentLengthValidationResult:
		logging.Warn("Different length validation result", types.Validation)
		simVal = 0
	case *DifferentTokensValidationResult:
		logging.Warn("Different tokens validation result", types.Validation)
		simVal = 0
	case *SimilarityValidationResult:
		simVal = result.(*SimilarityValidationResult).Value
		logging.Info("Cosine similarity validation result", types.Validation, "cosineSimValue", simVal)
	case *InvalidInferenceResult:
		simVal = 0
		logging.Warn("Invalid inference result", types.Validation, "reason", result.(*InvalidInferenceResult).Reason, "inferenceId", result.GetInferenceId(), "error", result.(*InvalidInferenceResult).Error)
	default:
		logging.Error("Unknown validation result type", types.Validation, "type", fmt.Sprintf("%T", result), "result", result)
		return nil, errors.New("unknown validation result type")
	}

	responseHash, _, err := utils.GetResponseHash(result.GetValidationResponseBytes())
	if err != nil {
		logging.Error("Failed to get response hash", types.Validation, "error", err)
		return nil, err
	}

	return &inference.MsgValidation{
		Id:              uuid.New().String(),
		InferenceId:     result.GetInferenceId(),
		ResponsePayload: string(result.GetValidationResponseBytes()),
		ResponseHash:    responseHash,
		Value:           simVal,
	}, nil
}
