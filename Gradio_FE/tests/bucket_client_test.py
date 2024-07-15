import requests

def send_audio(file_path):
    url = 'http://localhost:5000/audio/upload'
    files = {'audio': open(file_path, 'rb')}
    response = requests.post(url, files=files)
    return response.json()

# Example usage
if __name__ == '__main__':
    audio_file_path = '../wavfiles/E_Major_Guitar.wav'
    result = send_audio(audio_file_path)
    print(result)