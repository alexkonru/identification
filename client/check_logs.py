import grpc
import biometry_pb2
import biometry_pb2_grpc
import sys

def check_logs(address):
    print(f"Connecting to {address}...")
    try:
        channel = grpc.insecure_channel(address)
        stub = biometry_pb2_grpc.GatekeeperStub(channel)
        
        print("Calling GetLogs...")
        response = stub.GetLogs(biometry_pb2.GetLogsRequest(limit=10, offset=0))
        print(f"Found {len(response.logs)} logs.")
        print("✅ GetLogs Successful!")

    except grpc.RpcError as e:
        print(f"❌ RPC Error: {e}")

if __name__ == "__main__":
    check_logs('127.0.0.1:50051')
