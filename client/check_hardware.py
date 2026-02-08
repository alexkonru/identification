import grpc
import biometry_pb2
import biometry_pb2_grpc
import sys

def check_hardware(address):
    print(f"Connecting to {address}...")
    try:
        channel = grpc.insecure_channel(address)
        stub = biometry_pb2_grpc.GatekeeperStub(channel)
        
        # Check Hardware (Scan)
        print("Calling ScanHardware...")
        response = stub.ScanHardware(biometry_pb2.Empty())
        print(f"Found {len(response.found_devices)} devices.")
        for d in response.found_devices:
            print(f" - ID: {d.id}, Name: {d.name}, Type: {d.device_type}, Conn: {d.connection_string}")
        
        # Check Hardware (List) - this is what hardware-controller uses
        print("\nCalling ListDevices...")
        response_list = stub.ListDevices(biometry_pb2.ListDevicesRequest())
        print(f"Found {len(response_list.devices)} devices.")
        for d in response_list.devices:
            print(f" - ID: {d.id}, Name: {d.name}, Type: {d.device_type}, Conn: {d.connection_string}")

        # Check Audio Status via Gateway
        print("\nChecking System Status...")
        status = stub.GetSystemStatus(biometry_pb2.Empty())
        print(f"Audio Status: {status.audio.message} (Online: {status.audio.online})")
        print(f"Audio Device: {status.audio.device}")

    except grpc.RpcError as e:
        print(f"❌ RPC Error: {e}")

if __name__ == "__main__":
    check_hardware('127.0.0.1:50051')
