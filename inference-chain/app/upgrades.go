//go:build !upgraded

package app

import (
	"context"
	"fmt"

	upgradetypes "cosmossdk.io/x/upgrade/types"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/cosmos/cosmos-sdk/types/module"
	districutiontypes "github.com/cosmos/cosmos-sdk/x/distribution/types"
	slashingtypes "github.com/cosmos/cosmos-sdk/x/slashing/types"
	stakingtypes "github.com/cosmos/cosmos-sdk/x/staking/types"
	v0_2_2 "github.com/productscience/inference/app/upgrades/v0_2_2"
	v0_2_3 "github.com/productscience/inference/app/upgrades/v0_2_3"
	inferencetypes "github.com/productscience/inference/x/inference/types"
)

func CreateEmptyUpgradeHandler(
	mm *module.Manager,
	configurator module.Configurator) upgradetypes.UpgradeHandler {
	return func(ctx context.Context, plan upgradetypes.Plan, vm module.VersionMap) (module.VersionMap, error) {

		for moduleName, version := range vm {
			fmt.Printf("Module: %s, Version: %d\n", moduleName, version)
		}
		fmt.Printf("OrderMigrations: %v\n", mm.OrderMigrations)

		// For some reason, the capability module doesn't have a version set, but it DOES exist, causing
		// the `InitGenesis` to panic.
		if _, ok := vm["capability"]; !ok {
			vm["capability"] = mm.Modules["capability"].(module.HasConsensusVersion).ConsensusVersion()
		}
		return mm.RunMigrations(ctx, configurator, vm)
	}
}

func (app *App) setupUpgradeHandlers() {
	app.Logger().Info("Setting up upgrade handlers")
	upgradeInfo, err := app.UpgradeKeeper.ReadUpgradeInfoFromDisk()
	if err != nil {
		app.Logger().Error("Failed to read upgrade info from disk", "error", err)
		return
	}
	app.Logger().Info("Applying upgrade", "upgradeInfo", upgradeInfo)

	app.UpgradeKeeper.SetUpgradeHandler(v0_2_2.UpgradeName, v0_2_2.CreateUpgradeHandler(app.ModuleManager, app.Configurator(), app.InferenceKeeper))
	app.UpgradeKeeper.SetUpgradeHandler(v0_2_3.UpgradeName, v0_2_3.CreateUpgradeHandler(app.ModuleManager, app.Configurator(), app.InferenceKeeper))
}

func (app *App) registerMigrations() {
	app.Configurator().RegisterMigration(inferencetypes.ModuleName, 4, func(ctx sdk.Context) error {
		return nil
	})

	app.Configurator().RegisterMigration(inferencetypes.ModuleName, 5, func(ctx sdk.Context) error {
		return nil
	})

	app.Configurator().RegisterMigration(districutiontypes.ModuleName, 3, func(ctx sdk.Context) error {
		return nil
	})

	app.Configurator().RegisterMigration(slashingtypes.ModuleName, 4, func(ctx sdk.Context) error {
		return nil
	})

	app.Configurator().RegisterMigration(stakingtypes.ModuleName, 5, func(ctx sdk.Context) error {
		return nil
	})
}
