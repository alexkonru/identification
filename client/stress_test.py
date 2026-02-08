import grpc
import biometry_pb2
import biometry_pb2_grpc
import sys
import time

def stress_test(address):
    print(f"Connecting to {address}...")
    channel = grpc.insecure_channel(address)
    stub = biometry_pb2_grpc.GatekeeperStub(channel)
    
    for i in range(10):
        try:
            print(f"Request {i+1}...")
            stub.ListZones(biometry_pb2.ListZonesRequest())
            stub.ListRooms(biometry_pb2.ListRoomsRequest())
            stub.ListDevices(biometry_pb2.ListDevicesRequest())
        except grpc.RpcError as e:
            print(f"❌ RPC Error: {e}")
            break
    print("Done.")

if __name__ == "__main__":
    stress_test('127.0.0.1:50051')
