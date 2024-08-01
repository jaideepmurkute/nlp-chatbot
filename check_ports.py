import socket

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0`.0.0', port))
            return True
        except OSError:
            return False

port = 5000
if check_port(port):
    print(f"Port {port} is open and Flask is likely running.")
else:
    print(f"Port {port} is not open.")
