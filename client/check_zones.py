import grpc
import biometry_pb2
import biometry_pb2_grpc
import sys

def check_zones(address):
    print(f"Connecting to {address}...")
    try:
        channel = grpc.insecure_channel(address)
        stub = biometry_pb2_grpc.GatekeeperStub(channel)
        
        print("Calling ListZones...")
        response = stub.ListZones(biometry_pb2.ListZonesRequest())
        print(f"Found {len(response.zones)} zones.")
        print("✅ ListZones Successful!")

    except grpc.RpcError as e:
        print(f"❌ RPC Error: {e}")

if __name__ == "__main__":
    check_zones('127.0.0.1:50051')
