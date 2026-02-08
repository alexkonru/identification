import grpc
import biometry_pb2
import biometry_pb2_grpc
import sys

def test_connection(address):
    print(f"Testing connection to {address}...")
    try:
        channel = grpc.insecure_channel(address)
        stub = biometry_pb2_grpc.GatekeeperStub(channel)
        # Try a simple call (SystemStatus is good)
        response = stub.GetSystemStatus(biometry_pb2.Empty())
        print("✅ Connection Successful!")
        print(f"Response object: {response}")
        print(f"Gateway Status: {'Online' if response.gateway.online else 'Offline'}")
        print(f"Vision Status: {response.vision.message}")
    except grpc.RpcError as e:
        print(f"❌ Connection Failed: {e.code()}")
        print(f"Details: {e.details()}")

if __name__ == "__main__":
    addr = '127.0.0.1:50051'
    if len(sys.argv) > 1:
        addr = sys.argv[1]
    test_connection(addr)
