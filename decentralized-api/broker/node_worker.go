package broker

import (
	"context"
	"decentralized-api/logging"
	"decentralized-api/mlnodeclient"
	"sync"
	"time"

	"github.com/productscience/inference/x/inference/types"
)

type commandWithContext struct {
	cmd NodeWorkerCommand
	ctx context.Context
}

// NodeWorker handles asynchronous operations for a specific node
type NodeWorker struct {
	nodeId            string
	node              *NodeWithState
	mlClient          mlnodeclient.MLNodeClient
	clientMu          sync.RWMutex
	broker            *Broker
	commands          chan commandWithContext
	shutdown          chan struct{}
	wg                sync.WaitGroup
	availableVersions map[string]bool
	versionsMu        sync.Mutex
}

// NewNodeWorkerWithClient creates a new worker with a custom client (for testing)
func NewNodeWorkerWithClient(nodeId string, node *NodeWithState, client mlnodeclient.MLNodeClient, broker *Broker) *NodeWorker {
	worker := &NodeWorker{
		nodeId:            nodeId,
		node:              node,
		mlClient:          client,
		broker:            broker,
		commands:          make(chan commandWithContext, 10),
		availableVersions: make(map[string]bool),
		shutdown:          make(chan struct{}),
	}
	go worker.run()
	return worker
}

// run is the main event loop for the worker
func (w *NodeWorker) run() {
	for {
		select {
		case item := <-w.commands:
			result := item.cmd.Execute(item.ctx, w)

			// Queue a command back to the broker to update the state
			updateCmd := NewUpdateNodeResultCommand(w.nodeId, result)
			if err := w.broker.QueueMessage(updateCmd); err != nil {
				logging.Error("Failed to queue node result update command", types.Nodes,
					"node_id", w.nodeId, "error", err)
			}
			// We don't wait for the response from updateCmd, the worker's job is done.
			w.wg.Done()
		case <-w.shutdown:
			// Drain remaining commands before shutting down
			close(w.commands)
			for item := range w.commands {
				result := item.cmd.Execute(item.ctx, w)
				updateCmd := NewUpdateNodeResultCommand(w.nodeId, result)
				if err := w.broker.QueueMessage(updateCmd); err != nil {
					logging.Error("Failed to queue node result update command during shutdown", types.Nodes,
						"node_id", w.nodeId, "error", err)
				}
				w.wg.Done()
			}
			return
		}
	}
}

// Submit queues a command for execution on this node
func (w *NodeWorker) Submit(ctx context.Context, cmd NodeWorkerCommand) bool {
	w.wg.Add(1)
	select {
	case w.commands <- commandWithContext{cmd: cmd, ctx: ctx}:
		return true
	default:
		w.wg.Done()
		return false
	}
}

// Shutdown gracefully stops the worker
func (w *NodeWorker) Shutdown() {
	close(w.shutdown)
	w.wg.Wait() // Wait for all pending commands to complete
}

func (w *NodeWorker) RefreshClientImmediate(oldVersion, newVersion string) {
	w.clientMu.Lock()
	oldClient := w.mlClient
	w.mlClient = w.broker.NewNodeClient(&w.node.Node)
	w.clientMu.Unlock()

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		// TODO: should be /kill, not /stop
		if err := oldClient.Stop(ctx); err != nil {
			logging.Warn("Failed to stop old MLNode client during immediate version transition", types.Nodes,
				"node_id", w.nodeId, "oldVersion", oldVersion, "newVersion", newVersion, "error", err)
		} else {
			logging.Info("Successfully stopped old MLNode client during immediate version transition", types.Nodes,
				"node_id", w.nodeId, "oldVersion", oldVersion, "newVersion", newVersion)
		}
	}()

	logging.Info("Immediately refreshed MLNode client", types.Nodes,
		"node_id", w.nodeId, "oldVersion", oldVersion, "newVersion", newVersion)
}

// isVersionKnownAlive checks the cache to see if a version is already known to be alive.
// It returns true if it's cached and alive, false otherwise.
func (w *NodeWorker) isVersionKnownAlive(version string) bool {
	w.versionsMu.Lock()
	defer w.versionsMu.Unlock()
	if alive, ok := w.availableVersions[version]; ok && alive {
		return true
	}
	return false
}

func (w *NodeWorker) CheckClientVersionAlive(version string, factory mlnodeclient.ClientFactory) (bool, error) {
	if w.isVersionKnownAlive(version) {
		return true, nil
	}

	node := w.node.Node
	pocUrl := node.PoCUrlWithVersion(version)
	inferenceUrl := node.InferenceUrlWithVersion(version)

	versionClient := factory.CreateClient(pocUrl, inferenceUrl)
	_, err := versionClient.NodeState(context.Background())

	w.versionsMu.Lock()
	defer w.versionsMu.Unlock()
	if err != nil {
		w.availableVersions[version] = false
		return false, err
	}
	w.availableVersions[version] = true
	return true, nil
}

func (w *NodeWorker) GetClient() mlnodeclient.MLNodeClient {
	w.clientMu.RLock()
	defer w.clientMu.RUnlock()
	return w.mlClient
}

// NodeWorkGroup manages parallel execution across multiple node workers
type NodeWorkGroup struct {
	workers map[string]*NodeWorker
	mu      sync.RWMutex
}

// NewNodeWorkGroup creates a new work group
func NewNodeWorkGroup() *NodeWorkGroup {
	return &NodeWorkGroup{
		workers: make(map[string]*NodeWorker),
	}
}

// AddWorker adds a new worker to the group
func (g *NodeWorkGroup) AddWorker(nodeId string, worker *NodeWorker) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.workers[nodeId] = worker
}

// RemoveWorker removes and shuts down a worker
func (g *NodeWorkGroup) RemoveWorker(nodeId string) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if worker, exists := g.workers[nodeId]; exists {
		worker.Shutdown()
		delete(g.workers, nodeId)
	}
}

// GetWorker returns a specific worker (useful for node-specific commands)
func (g *NodeWorkGroup) GetWorker(nodeId string) (*NodeWorker, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	worker, exists := g.workers[nodeId]
	return worker, exists
}
