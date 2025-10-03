package poc

import (
	"decentralized-api/apiconfig"
	"decentralized-api/cosmosclient"
	"decentralized-api/logging"
	"encoding/binary"
	"encoding/hex"

	"github.com/productscience/inference/api/inference/inference"
	"github.com/productscience/inference/x/inference/types"
)

type RandomSeedManager interface {
	GenerateSeedInfo(epochIndex uint64)
	GetSeedForEpoch(epochIndex uint64) apiconfig.SeedInfo
	CreateNewSeed(epochIndex uint64) (*apiconfig.SeedInfo, error)
	ChangeCurrentSeed()
	RequestMoney(epochIndex uint64)
}

type RandomSeedManagerImpl struct {
	transactionRecorder cosmosclient.CosmosMessageClient
	configManager       *apiconfig.ConfigManager
}

func NewRandomSeedManager(
	transactionRecorder cosmosclient.CosmosMessageClient,
	configManager *apiconfig.ConfigManager,
) *RandomSeedManagerImpl {
	return &RandomSeedManagerImpl{
		transactionRecorder: transactionRecorder,
		configManager:       configManager,
	}
}

func (rsm *RandomSeedManagerImpl) GenerateSeedInfo(epochIndex uint64) {
	logging.Debug("Old Seed Signature", types.Claims, rsm.configManager.GetCurrentSeed())
	newSeed, err := rsm.CreateNewSeed(epochIndex)
	if err != nil {
		logging.Error("Failed to get next seed signature", types.Claims, "error", err)
		return
	}
	err = rsm.configManager.SetUpcomingSeed(*newSeed)
	if err != nil {
		logging.Error("Failed to set upcoming seed", types.Claims, "error", err)
		return
	}
	logging.Debug("New Seed Signature", types.Claims, "seed", rsm.configManager.GetUpcomingSeed())

	err = rsm.transactionRecorder.SubmitSeed(&inference.MsgSubmitSeed{
		EpochIndex: rsm.configManager.GetUpcomingSeed().EpochIndex,
		Signature:  rsm.configManager.GetUpcomingSeed().Signature,
	})
	if err != nil {
		logging.Error("Failed to send SubmitSeed transaction", types.Claims, "error", err)
	}
}

func (rsm *RandomSeedManagerImpl) ChangeCurrentSeed() {
	configManager := rsm.configManager
	err := configManager.SetPreviousSeed(configManager.GetCurrentSeed())
	if err != nil {
		logging.Error("Failed to set previous seed", types.Claims, "error", err)
		return
	}
	err = configManager.SetCurrentSeed(configManager.GetUpcomingSeed())
	if err != nil {
		logging.Error("Failed to set current seed", types.Claims, "error", err)
		return
	}
	err = configManager.SetUpcomingSeed(apiconfig.SeedInfo{})
	if err != nil {
		logging.Error("Failed to set upcoming seed", types.Claims, "error", err)
		return
	}
}

func (rsm *RandomSeedManagerImpl) GetSeedForEpoch(epochIndex uint64) apiconfig.SeedInfo {
	previousSeed := rsm.configManager.GetPreviousSeed()
	if previousSeed.EpochIndex == epochIndex && previousSeed.Seed != 0 {
		return previousSeed
	}

	seed, err := rsm.CreateNewSeed(epochIndex)
	if err != nil {
		logging.Error("Failed to create new seed", types.Claims, "error", err)
		return apiconfig.SeedInfo{}
	}
	return *seed
}

func (rsm *RandomSeedManagerImpl) RequestMoney(epochIndex uint64) {
	// FIXME: we can also imagine a scenario where we weren't updating the seed for a few epochs
	//  e.g. generation fails a few times in a row for some reason
	//  Solution: query seed here?
	seed := rsm.GetSeedForEpoch(epochIndex)

	logging.Info("IsSetNewValidatorsStage: sending ClaimRewards transaction", types.Claims, "seed", seed)
	err := rsm.transactionRecorder.ClaimRewards(&inference.MsgClaimRewards{
		Seed:       seed.Seed,
		EpochIndex: seed.EpochIndex,
	})
	if err != nil {
		logging.Error("Failed to send ClaimRewards transaction", types.Claims, "error", err)
	}
}

func (rsm *RandomSeedManagerImpl) CreateNewSeed(epochIndex uint64) (*apiconfig.SeedInfo, error) {
	newSeed, err := rsm.createSeedForEpoch(epochIndex)
	if err != nil {
		logging.Error("Failed to get seedBytes", types.Claims, "error", err)
		return nil, err
	}

	// Encode seed for signing
	seedBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(seedBytes, uint64(newSeed))

	signature, err := rsm.transactionRecorder.SignBytes(seedBytes)
	if err != nil {
		logging.Error("Failed to sign bytes", types.Claims, "error", err)
		return nil, err
	}

	return &apiconfig.SeedInfo{
		Seed:       newSeed,
		EpochIndex: epochIndex,
		Signature:  hex.EncodeToString(signature),
	}, nil
}

func (rsm *RandomSeedManagerImpl) createSeedForEpoch(epoch uint64) (int64, error) {
	initialSeedBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(initialSeedBytes, epoch)

	signed, err := rsm.transactionRecorder.SignBytes(initialSeedBytes)
	if err != nil {
		logging.Error("Failed to sign bytes", types.Claims, "error", err)
		return 0, err
	}

	signed8bytes := signed[:8]
	newSeed := int64(binary.BigEndian.Uint64(signed8bytes[:]) & ((1 << 63) - 1))
	if newSeed == 0 {
		newSeed = 1
	}

	return newSeed, nil
}
