/*Copyright 2021 IBM All Rights Reserved.

SPDX-License-Identifier: Apache-2.0
*/

package main

import (
	"bytes"
	"context"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"time"

	"github.com/hyperledger/fabric-gateway/pkg/client"
	"github.com/hyperledger/fabric-gateway/pkg/identity"
	"github.com/hyperledger/fabric-protos-go-apiv2/gateway"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
)

const (
	mspID        = "Org1MSP"
	cryptoPath   = "../../test-network/organizations/peerOrganizations/org1.example.com"
	certPath     = cryptoPath + "/users/User1@org1.example.com/msp/signcerts/cert.pem"
	keyPath      = cryptoPath + "/users/User1@org1.example.com/msp/keystore/"
	tlsCertPath  = cryptoPath + "/peers/peer0.org1.example.com/tls/ca.crt"
	peerEndpoint = "localhost:7051"
	gatewayPeer  = "peer0.org1.example.com"
)

var now = time.Now()
var assetId = fmt.Sprintf("asset%d", now.Unix()*1e3+int64(now.Nanosecond())/1e6)
var contract *client.Contract

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

// newGrpcConnection creates a gRPC connection to the Gateway server.
func newGrpcConnection() *grpc.ClientConn {
	certificate, err := loadCertificate(tlsCertPath)
	if err != nil {
		panic(err)
	}

	certPool := x509.NewCertPool()
	certPool.AddCert(certificate)
	transportCredentials := credentials.NewClientTLSFromCert(certPool, gatewayPeer)

	connection, err := grpc.Dial(peerEndpoint, grpc.WithTransportCredentials(transportCredentials))
	if err != nil {
		panic(fmt.Errorf("failed to create gRPC connection: %w", err))
	}

	return connection
}

// newIdentity creates a client identity for this Gateway connection using an X.509 certificate.
func newIdentity() *identity.X509Identity {
	certificate, err := loadCertificate(certPath)
	if err != nil {
		panic(err)
	}

	id, err := identity.NewX509Identity(mspID, certificate)
	if err != nil {
		panic(err)
	}

	return id
}

func loadCertificate(filename string) (*x509.Certificate, error) {
	certificatePEM, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read certificate file: %w", err)
	}
	return identity.CertificateFromPEM(certificatePEM)
}

// newSign creates a function that generates a digital signature from a message digest using a private key.
func newSign() identity.Sign {
	files, err := os.ReadDir(keyPath)
	if err != nil {
		panic(fmt.Errorf("failed to read private key directory: %w", err))
	}
	privateKeyPEM, err := os.ReadFile(path.Join(keyPath, files[0].Name()))

	if err != nil {
		panic(fmt.Errorf("failed to read private key file: %w", err))
	}

	privateKey, err := identity.PrivateKeyFromPEM(privateKeyPEM)
	if err != nil {
		panic(err)
	}

	sign, err := identity.NewPrivateKeySign(privateKey)
	if err != nil {
		panic(err)
	}

	return sign
}
func main() {

	http.HandleFunc("/data", postdata)
	http.HandleFunc("/updaterep", chrep)
	http.HandleFunc("/current", queryall)
	http.HandleFunc("/data_state", querydata)
	http.HandleFunc("/rep_state", queryrep)
	// The gRPC client connection should be shared by all Gateway connections to this endpoint
	clientConnection := newGrpcConnection()
	defer clientConnection.Close()

	id := newIdentity()
	sign := newSign()

	// Create a Gateway connection for a specific client identity
	gw, err := client.Connect(
		id,
		client.WithSign(sign),
		client.WithClientConnection(clientConnection),
		// Default timeouts for different gRPC calls
		client.WithEvaluateTimeout(5*time.Second),
		client.WithEndorseTimeout(15*time.Second),
		client.WithSubmitTimeout(5*time.Second),
		client.WithCommitStatusTimeout(1*time.Minute),
	)
	if err != nil {
		panic(err)
	}
	defer gw.Close()

	// Override default values for chaincode and channel name as they may differ in testing contexts.
	chaincodeName := "basic"
	if ccname := os.Getenv("CHAINCODE_NAME"); ccname != "" {
		chaincodeName = ccname
	}

	channelName := "mychannel"
	if cname := os.Getenv("CHANNEL_NAME"); cname != "" {
		channelName = cname
	}

	network := gw.GetNetwork(channelName)
	contract = network.GetContract(chaincodeName)
	herr := http.ListenAndServe(":3333", nil)
	if errors.Is(herr, http.ErrServerClosed) {
		fmt.Printf("server closed\n")
	} else if herr != nil {
		fmt.Printf("error starting server: %s\n", herr)
		os.Exit(1)
	}
	/*router := gin.Default()
	router.POST("/data", postdata)

	router.Run("localhost:8090")*/

	/*readAssetByID(contract)
	initLedger()
	transferAssetAsync(contract)
	exampleErrorHandling(contract)*/
}

func postdata(w http.ResponseWriter, r *http.Request) {
	var newblock Data_block
	var now = time.Now()
	newblock.ID = fmt.Sprintf("block%d", now.Unix()*1e3+int64(now.Nanosecond())/1e6)
	newblock.EdgeServer = r.PostFormValue("edgeserver")
	newblock.Vehicle = r.PostFormValue("vehicle")
	newblock.Model = r.PostFormValue("model")
	newblock.BlockHash = r.PostFormValue("blockhash")
	createAsset(newblock)

}
func chrep(w http.ResponseWriter, r *http.Request) {
	_id := r.PostFormValue("id")
	updaterep(_id)

}
func querydata(w http.ResponseWriter, r *http.Request) {
	state := getAlldatablocks()
	io.WriteString(w, state)
}
func queryrep(w http.ResponseWriter, r *http.Request) {
	state := getAllrepute()
	io.WriteString(w, state)
}
func queryall(w http.ResponseWriter, r *http.Request) {
	state := getAllAssets()
	io.WriteString(w, state)
}

// This type of transaction would typically only be run once by an application the first time it was started after its
// initial deployment. A new version of the chaincode deployed later would likely not need to run an "init" function.
func initLedger() {
	fmt.Printf("\n--> Submit Transaction: InitLedger, function creates the initial set of assets on the ledger \n")

	_, err := contract.SubmitTransaction("InitLedger")
	if err != nil {
		fmt.Printf("*** Transaction Failed\n")
	}

	fmt.Printf("*** Transaction committed successfully\n")
}
func getAlldatablocks() string {
	fmt.Println("\n--> Evaluate Transaction: GetAllAssets, function returns all the current assets on the ledger")

	evaluateResult, err := contract.EvaluateTransaction("GetAllDatablocks")
	if err != nil {
		fmt.Printf("*** Transaction Failed\n")
	}
	result := formatJSON(evaluateResult)
	return result
}
func getAllrepute() string {
	fmt.Println("\n--> Evaluate Transaction: GetAllAssets, function returns all the current assets on the ledger")

	evaluateResult, err := contract.EvaluateTransaction("GetAllReputation")
	if err != nil {
		panic(fmt.Errorf("failed to evaluate transaction: %w", err))
	}
	result := formatJSON(evaluateResult)
	return result
}

// Evaluate a transaction to query ledger state.
func getAllAssets() string {
	fmt.Println("\n--> Evaluate Transaction: GetAllAssets, function returns all the current assets on the ledger")

	evaluateResult, err := contract.EvaluateTransaction("GetAllAssets")
	if err != nil {
		panic(fmt.Errorf("failed to evaluate transaction: %w", err))
	}
	result := formatJSON(evaluateResult)
	return result
}

// Submit a transaction synchronously, blocking until it has been committed to the ledger.
func createAsset(newAsset Data_block) {
	fmt.Printf("\n--> Submit Transaction: CreateAsset, creates new DataBlock \n")
	rep := "10"

	_, err := contract.SubmitTransaction("CreateData_block", newAsset.ID, newAsset.EdgeServer, newAsset.Vehicle, newAsset.Model, newAsset.BlockHash, rep)

	if err != nil {
		fmt.Printf("failed to submit transaction\n")
		print(fmt.Errorf("failed to submit transaction: %w", err))
	} else {
		fmt.Printf("*** Transaction committed successfully\n")
	}

}
func updaterep(_id string) {
	fmt.Printf("\n--> Submit Transaction: Update Reputation, Updates Reputation \n")
	print(_id)
	_, err := contract.SubmitTransaction("DecRepute", _id)
	if err != nil {
		print(fmt.Errorf("failed to submit transaction: %w", err))
	} else {
		fmt.Printf("*** Transaction committed successfully\n")
	}

}
func getrep(_id string) []byte {
	fmt.Printf("\n--> Evaluate Transaction: ReadAsset, function returns asset attributes\n")

	evaluateResult, err := contract.EvaluateTransaction("GetRepute", _id)
	if err != nil {
		panic(fmt.Errorf("failed to evaluate transaction: %w", err))
	}

	return evaluateResult
}
func readrep(contract *client.Contract, _id string) string {
	fmt.Printf("\n--> Evaluate Transaction: ReadAsset, function returns asset attributes\n")

	evaluateResult, err := contract.EvaluateTransaction("ReadRepute", _id)
	if err != nil {
		panic(fmt.Errorf("failed to evaluate transaction: %w", err))
	}
	result := formatJSON(evaluateResult)

	return result
}

// Evaluate a transaction by assetID to query ledger state.
func readblock(contract *client.Contract, _id string) string {
	fmt.Printf("\n--> Evaluate Transaction: ReadAsset, function returns asset attributes\n")

	evaluateResult, err := contract.EvaluateTransaction("ReadData_block", _id)
	if err != nil {
		panic(fmt.Errorf("failed to evaluate transaction: %w", err))
	}
	result := formatJSON(evaluateResult)

	return result
}

// Submit transaction asynchronously, blocking until the transaction has been sent to the orderer, and allowing
// this thread to process the chaincode response (e.g. update a UI) without waiting for the commit notification
func transferAssetAsync(contract *client.Contract) {
	fmt.Printf("\n--> Async Submit Transaction: TransferAsset, updates existing asset owner")

	submitResult, commit, err := contract.SubmitAsync("TransferAsset", client.WithArguments(assetId, "Mark"))
	if err != nil {
		panic(fmt.Errorf("failed to submit transaction asynchronously: %w", err))
	}

	fmt.Printf("\n*** Successfully submitted transaction to transfer ownership from %s to Mark. \n", string(submitResult))
	fmt.Println("*** Waiting for transaction commit.")

	if commitStatus, err := commit.Status(); err != nil {
		panic(fmt.Errorf("failed to get commit status: %w", err))
	} else if !commitStatus.Successful {
		panic(fmt.Errorf("transaction %s failed to commit with status: %d", commitStatus.TransactionID, int32(commitStatus.Code)))
	}

	fmt.Printf("*** Transaction committed successfully\n")
}

// Submit transaction, passing in the wrong number of arguments ,expected to throw an error containing details of any error responses from the smart contract.
func exampleErrorHandling(contract *client.Contract) {
	fmt.Println("\n--> Submit Transaction: UpdateAsset asset70, asset70 does not exist and should return an error")

	_, err := contract.SubmitTransaction("UpdateAsset", "asset70", "blue", "5", "Tomoko", "300")
	if err == nil {
		panic("******** FAILED to return an error")
	}

	fmt.Println("*** Successfully caught the error:")

	switch err := err.(type) {
	case *client.EndorseError:
		fmt.Printf("Endorse error for transaction %s with gRPC status %v: %s\n", err.TransactionID, status.Code(err), err)
	case *client.SubmitError:
		fmt.Printf("Submit error for transaction %s with gRPC status %v: %s\n", err.TransactionID, status.Code(err), err)
	case *client.CommitStatusError:
		if errors.Is(err, context.DeadlineExceeded) {
			fmt.Printf("Timeout waiting for transaction %s commit status: %s", err.TransactionID, err)
		} else {
			fmt.Printf("Error obtaining commit status for transaction %s with gRPC status %v: %s\n", err.TransactionID, status.Code(err), err)
		}
	case *client.CommitError:
		fmt.Printf("Transaction %s failed to commit with status %d: %s\n", err.TransactionID, int32(err.Code), err)
	default:
		panic(fmt.Errorf("unexpected error type %T: %w", err, err))
	}

	// Any error that originates from a peer or orderer node external to the gateway will have its details
	// embedded within the gRPC status error. The following code shows how to extract that.
	statusErr := status.Convert(err)

	details := statusErr.Details()
	if len(details) > 0 {
		fmt.Println("Error Details:")

		for _, detail := range details {
			switch detail := detail.(type) {
			case *gateway.ErrorDetail:
				fmt.Printf("- address: %s, mspId: %s, message: %s\n", detail.Address, detail.MspId, detail.Message)
			}
		}
	}
}

// Format JSON data
func formatJSON(data []byte) string {
	var prettyJSON bytes.Buffer
	if err := json.Indent(&prettyJSON, data, "", "  "); err != nil {
		panic(fmt.Errorf("failed to parse JSON: %w", err))
	}
	return prettyJSON.String()
}
