package chaincode

import (
	"encoding/json"
	"fmt"
	"regexp"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// SmartContract provides functions for managing an Asset
type SmartContract struct {
	contractapi.Contract
}

// Asset describes basic details of what makes up a simple asset
// Insert struct field in alphabetic order => to achieve determinism across languages
// golang keeps the order when marshal to json but doesn't order automatically
/*type Asset struct {
	AppraisedValue int    `json:"AppraisedValue"`
	Color          string `json:"Color"`
	ID             string `json:"ID"`
	Owner          string `json:"Owner"`
	Size           int    `json:"Size"`
}*/
type Data_block struct {
	ID         string `json:"ID"`
	EdgeServer string `json:"EdgeServer"`
	Vehicle    string `json:"Vehicle"`
	BlockHash  string `json:"BlockHash"`
	Model      string `json:"Model"`
}
type Repute struct {
	ID         string `json:"ID"`
	Reputation int    `json:"Reputation"`
}
type _Asset struct {
	ID         string `json:"ID"`
	EdgeServer string `json:"EdgeServer"`
	BlockHash  string `json:"BlockHash"`
	Vehicle    string `json:"Vehicle"`
	Model      string `json:"Model"`
	Reputation int    `json:"Reputation"`
}

// InitLedger adds a base set of assets to the ledger
func (s *SmartContract) InitLedger(ctx contractapi.TransactionContextInterface) error {
	/*	assets := []Asset{
		{ID: "asset1", Color: "blue", Size: 5, Owner: "Tomoko", AppraisedValue: 300},
		{ID: "asset2", Color: "red", Size: 5, Owner: "Brad", AppraisedValue: 400},
		{ID: "asset3", Color: "green", Size: 10, Owner: "Jin Soo", AppraisedValue: 500},
		{ID: "asset4", Color: "yellow", Size: 10, Owner: "Max", AppraisedValue: 600},
		{ID: "asset5", Color: "black", Size: 15, Owner: "Adriana", AppraisedValue: 700},
		{ID: "asset6", Color: "white", Size: 15, Owner: "Michel", AppraisedValue: 800},
	}*/
	assets := []Repute{
		{ID: "ev1", Reputation: 5},
		{ID: "ev2", Reputation: 5},
		{ID: "ev3", Reputation: 5},
		{ID: "ev4", Reputation: 5},
		{ID: "ev5", Reputation: 5},
		{ID: "ev6", Reputation: 5},
		{ID: "ev7", Reputation: 5},
		{ID: "ev8", Reputation: 5},
		{ID: "ev9", Reputation: 5},
		{ID: "rsu1", Reputation: 5},
		{ID: "rsu2", Reputation: 5},
	}

	for _, asset := range assets {
		assetJSON, err := json.Marshal(asset)
		if err != nil {
			return err
		}

		err = ctx.GetStub().PutState(asset.ID, assetJSON)
		if err != nil {
			return fmt.Errorf("failed to put to world state. %v", err)
		}
	}

	return nil
}

// AssetExists returns true when asset with given ID exists in world state
func (s *SmartContract) AssetExists(ctx contractapi.TransactionContextInterface, id string) (bool, error) {
	assetJSON, err := ctx.GetStub().GetState(id)
	if err != nil {
		return false, fmt.Errorf("failed to read from world state: %v", err)
	}

	return assetJSON != nil, nil
}

// CreateData_block issues a new datablock to the world state with given details.
func (s *SmartContract) CreateData_block(ctx contractapi.TransactionContextInterface, id string, edgeserver string, vehicle string, model string, blockhash string) error {
	ev_rep, _ := s.GetRepute(ctx, vehicle)
	rsu_rep, _ := s.GetRepute(ctx, edgeserver)
	_blockrep := ev_rep + rsu_rep
	if _blockrep < 5 {
		return fmt.Errorf("block reputation too low")
	}
	exists, err := s.AssetExists(ctx, id)
	if err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("the asset %s already exists", id)
	}

	asset := Data_block{
		ID:         id,
		EdgeServer: edgeserver,
		BlockHash:  blockhash,
		Vehicle:    vehicle,
		Model:      model,
	}
	assetJSON, err := json.Marshal(asset)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(id, assetJSON)
}
func (s *SmartContract) CreateRepute(ctx contractapi.TransactionContextInterface, id string) error {
	exists, err := s.AssetExists(ctx, id)
	if err != nil {
		return err
	}
	if exists {
		return fmt.Errorf("the asset %s already exists", id)
	}
	asset := Repute{
		ID:         id,
		Reputation: 5,
	}
	assetJSON, err := json.Marshal(asset)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(id, assetJSON)
}

// ReadAsset returns the asset stored in the world state with given id.
func (s *SmartContract) ReadData_block(ctx contractapi.TransactionContextInterface, id string) (*Data_block, error) {
	match, _ := regexp.MatchString("(block[0-9]*)", id)
	if !match {
		return nil, fmt.Errorf("not a valid blockid")
	}
	assetJSON, err := ctx.GetStub().GetState(id)
	if err != nil {
		return nil, fmt.Errorf("failed to read from world state: %v", err)
	}
	if assetJSON == nil {
		return nil, fmt.Errorf("the asset %s does not exist", id)
	}

	var db Data_block
	err = json.Unmarshal(assetJSON, &db)
	if err != nil {
		return nil, err
	}
	return &db, nil
}
func (s *SmartContract) ReadRepute(ctx contractapi.TransactionContextInterface, id string) (*Repute, error) {
	match, _ := regexp.MatchString("((ev[0-9]*)|(rsu[0-9]*))", id)
	if !match {
		return nil, fmt.Errorf("not a valid reputationid")
	}
	assetJSON, err := ctx.GetStub().GetState(id)
	if err != nil {
		return nil, fmt.Errorf("failed to read from world state: %v", err)
	}
	if assetJSON == nil {
		return nil, fmt.Errorf("the asset %s does not exist", id)
	}
	var rep Repute
	err = json.Unmarshal(assetJSON, &rep)
	if err != nil {
		return nil, err
	}
	return &rep, nil
}
func (s *SmartContract) DecRepute(ctx contractapi.TransactionContextInterface, _id string) error {
	exists, err := s.AssetExists(ctx, _id)
	if err != nil {
		return err
	}
	if !exists {
		return fmt.Errorf("the asset %s does not exist", _id)
	}
	_rep, _ := s.GetRepute(ctx, _id)
	if _rep > 0 {
		_rep = _rep - 1
	}
	// overwriting original asset with new asset
	asset := Repute{
		ID:         _id,
		Reputation: _rep,
	}
	assetJSON, err := json.Marshal(asset)
	if err != nil {
		return err
	}

	return ctx.GetStub().PutState(_id, assetJSON)
}
func (s *SmartContract) GetRepute(ctx contractapi.TransactionContextInterface, _id string) (int, error) {
	match, _ := regexp.MatchString("((ev[0-9]*)|(rsu[0-9]*))", _id)
	if !match {
		return 99999, fmt.Errorf("not a valid reputationid")
	}
	assetJSON, err := ctx.GetStub().GetState(_id)
	if err != nil {
		return 99999, fmt.Errorf("failed to read from world state: %v", err)
	}
	if assetJSON == nil {
		return 99999, fmt.Errorf("the asset %s does not exist", _id)
	}
	var rep Repute
	err = json.Unmarshal(assetJSON, &rep)
	if err != nil {
		return 99999, err
	}
	return rep.Reputation, nil
}
func (s *SmartContract) GetAllDatablocks(ctx contractapi.TransactionContextInterface) ([]*Data_block, error) {
	// range query with empty string for startKey and endKey does an
	// open-ended query of all assets in the chaincode namespace.
	resultsIterator, err := ctx.GetStub().GetStateByRange("", "")
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()
	var record []*Data_block
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {

			return nil, err
		}
		var db Data_block
		err = json.Unmarshal(queryResponse.Value, &db)
		if err == nil {
			record = append(record, &db)
		}

	}

	return record, nil
}
func (s *SmartContract) GetAllReputation(ctx contractapi.TransactionContextInterface) ([]*Repute, error) {
	// range query with empty string for startKey and endKey does an
	// open-ended query of all assets in the chaincode namespace.
	resultsIterator, err := ctx.GetStub().GetStateByRange("", "")
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()
	var record []*Repute
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}
		var rep Repute
		err = json.Unmarshal(queryResponse.Value, &rep)
		if err == nil {
			record = append(record, &rep)
		}

	}

	return record, nil
}

// GetAllAssets returns all assets found in world state
func (s *SmartContract) GetAllAssets(ctx contractapi.TransactionContextInterface) ([]*_Asset, error) {
	// range query with empty string for startKey and endKey does an
	// open-ended query of all assets in the chaincode namespace.
	resultsIterator, err := ctx.GetStub().GetStateByRange("", "")
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()
	var record []*_Asset
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}

		var db Data_block
		var rep Repute
		err = json.Unmarshal(queryResponse.Value, &db)
		if err == nil {
			asset := _Asset{
				ID:         db.ID,
				Reputation: 0,
				BlockHash:  db.BlockHash,
				EdgeServer: db.EdgeServer,
				Model:      db.Model,
				Vehicle:    db.Vehicle,
			}
			record = append(record, &asset)
		}
		err = json.Unmarshal(queryResponse.Value, &rep)
		if err == nil {
			asset := _Asset{
				ID:         rep.ID,
				Reputation: rep.Reputation,
				BlockHash:  "",
				EdgeServer: "",
				Model:      "",
				Vehicle:    "",
			}
			record = append(record, &asset)
		}

	}

	return record, nil
}

/*








// UpdateAsset updates an existing asset in the world state with provided parameters.


// DeleteAsset deletes an given asset from the world state.
/*func (s *SmartContract) DeleteAsset(ctx contractapi.TransactionContextInterface, id string) error {
	exists, err := s.AssetExists(ctx, id)
	if err != nil {
		return err
	}
	if !exists {
		return fmt.Errorf("the asset %s does not exist", id)
	}

	return ctx.GetStub().DelState(id)
}



// TransferAsset updates the owner field of asset with given id in world state, and returns the old owner.
/*func (s *SmartContract) TransferAsset(ctx contractapi.TransactionContextInterface, id string, newOwner string) (string, error) {
	asset, err := s.ReadAsset(ctx, id)
	if err != nil {
		return "", err
	}

	oldOwner := asset.Owner
	asset.Owner = newOwner

	assetJSON, err := json.Marshal(asset)
	if err != nil {
		return "", err
	}

	err = ctx.GetStub().PutState(id, assetJSON)
	if err != nil {
		return "", err
	}

	return oldOwner, nil
}


*/
