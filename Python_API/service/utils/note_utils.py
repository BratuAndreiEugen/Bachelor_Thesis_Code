import os
import re
import time
from io import BytesIO

import boto3
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm
import cv2

from service.utils.folder_utils import FolderUtils


class NoteUtils:
    def __init__(self, resolution=(1920, 1080), freq_min=10, freq_max=1000, image_folder="content"):
        self.__note_mappings = {
            "C": 0,
            "C#": 1,
            "D": 2,
            "D#": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "G": 7,
            "G#": 8,
            "A": 9,
            "A#": 10,
            "B": 11
        }
        self.__note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.__resolution = resolution
        self.__freq_min = freq_min
        self.__freq_max = freq_max
        self.__image_folder = image_folder

    def do_it_all(self, audio_file_path, fft_window_seconds, fps, instrument_pred=None, genre_pred=None):
        audio, sampling_rate = librosa.load(audio_file_path)
        print(audio.shape)
        print(sampling_rate)
        fft_window_size = int(sampling_rate * fft_window_seconds)

        note_dict = self.make_content_and_note_dict(fft_window_size, audio, fps=fps,
                                                                 sampling_rate=sampling_rate, instrument_predictions=instrument_pred, genre_predictions=genre_pred)
        sorted_note_dict = self.sort_note_dict(note_dict)
        pentatonic_scale = self.pentatonic_scale_finder(sorted_note_dict)
        scale = self.scale_finder(sorted_note_dict)
        local_video_path = os.path.join(self.__image_folder, "video.mp4")
        self.create_video(local_video_path)
        return pentatonic_scale, scale, local_video_path

    @staticmethod
    def freq_to_number(f):
        return 69 + 12 * np.log2(f / 440.0)

    @staticmethod
    def number_to_freq(n):
        return 440 * 2.0 ** ((n - 69) / 12.0)

    def note_name(self, n):
        return self.__note_names[n % 12] + str(int(n / 12 - 1))

    def plot_fft_matplotlib(self, p, xf, notes, instrument, genre):
        fig, ax = plt.subplots(figsize=(self.__resolution[0] / 100, self.__resolution[1] / 100), dpi=100)
        ax.plot(xf, p)
        ax.set_xlim(self.__freq_min, self.__freq_max)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Frequency (note)")
        ax.set_ylabel("Magnitude")
        title_string = ""
        if instrument:
            title_string = title_string + instrument + "  "
        if genre:
            title_string = title_string + genre + "  "
        ax.set_title(title_string)
        ax.grid(True)

        for note in notes:
            ax.text(note[0] + 10, note[2], note[1], fontsize=24)

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close(fig)
        return buffer

    def extract_sample(self, audio, frame_number, frame_offset, fft_window_size):
        end = frame_number * frame_offset
        begin = int(end - fft_window_size)

        if end == 0:
            # We have no audio yet, return all zeros (very beginning)
            return np.zeros((np.abs(begin)), dtype=float)
        elif begin < 0:
            # We have some audio, padd with zeros
            return np.concatenate([np.zeros((np.abs(begin)), dtype=float), audio[0:end]])
        else:
            # Usually this happens, return the next sample
            return audio[begin:end]

    def find_top_notes(self, fft, xf, num):
        if np.max(fft.real) < 0.001:
            return []

        lst = [x for x in enumerate(fft.real)]
        lst = sorted(lst, key=lambda x: x[1], reverse=True)

        idx = 0
        found = []
        found_note = set()
        while (idx < len(lst)) and (len(found) < num):
            f = xf[lst[idx][0]]
            y = lst[idx][1]
            n = self.freq_to_number(f)
            n0 = int(round(n))
            name = self.note_name(n0)

            if name not in found_note:
                found_note.add(name)
                s = [f, self.note_name(n0), y]
                found.append(s)
            idx += 1

        return found

    def make_content_and_note_dict(self, fft_window_size, audio, fps, sampling_rate, instrument_predictions=None, genre_predictions=None, instrument_predicion_step=3, genre_prediction_step=3):
        FolderUtils.clear_folder(self.__image_folder) # clear the folder with the previously generated images
        # Hanning window function
        window = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, fft_window_size, False)))

        xf = np.fft.rfftfreq(fft_window_size, 1 / sampling_rate)
        frame_count = int(int((len(audio)/sampling_rate)) * fps)
        frame_offset = int(len(audio) / frame_count)

        # Pass 1, find out the maximum amplitude so we can scale.
        mx = 0
        for frame_number in range(frame_count):
            sample = self.extract_sample(audio, frame_number, frame_offset, fft_window_size)

            fft = np.fft.rfft(sample * window)
            fft = np.abs(fft).real
            mx = max(np.max(fft), mx)

        print(f"Max amplitude: {mx}")

        note_dict = {}
        for note in self.__note_names:
            note_dict[note] = 0

        # Pass 2, produce the animation
        for frame_number in tqdm.tqdm(range(frame_count)):
            sample = self.extract_sample(audio, frame_number, frame_offset, fft_window_size)

            fft = np.fft.rfft(sample * window)
            fft = np.abs(fft) / mx

            s = self.find_top_notes(fft, xf, 3)
            # print(s)
            for note_data in s:
                if note_data[0] > 26:  # no point in reading lower notes really since they are not used in music
                    note_dict[re.sub(r'\d+', '', note_data[1])] = note_dict[re.sub(r'\d+', '', note_data[1])] + 1

            try:
                index_i = int((frame_number/fps)/instrument_predicion_step)
                frame_instrument_predict = instrument_predictions[index_i] # frame_number/fps gives the second in which this predict is shown and the further division extracts the index from the predict list
            except:
                frame_instrument_predict = ""
            try:
                index_g = int((frame_number/fps)/genre_prediction_step)
                frame_genre_predict = genre_predictions[index_g]
            except:
                frame_genre_predict = ""

            fig = self.plot_fft_matplotlib(fft.real, xf, s, frame_instrument_predict, frame_genre_predict)
            with open(f"{self.__image_folder}/frame{frame_number}.png", "wb") as img:
                img.write(fig.getbuffer())

        return note_dict

    def create_video(self, video_name='video.mp4', fps=30):
        images = [img for img in os.listdir(self.__image_folder) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(self.__image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'X264'), fps, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(self.__image_folder, image)))

        cv2.destroyAllWindows()
        video.release()

    @staticmethod
    def sort_note_dict(note_dict):
        sorted_items = sorted(note_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_items

    def scale_finder(self, sorted_items):
        note_list = sorted_items[:7]  # notes in a list
        print(note_list)
        current_note = note_list[0]
        ordered_scale = [current_note]
        while len(ordered_scale) < 7:
            min_int = 4
            next_note = current_note
            for elem in note_list:
                if (abs(self.__note_mappings[elem[0]] - self.__note_mappings[current_note[0]]) < min_int and
                    self.__note_mappings[elem[0]] -
                    self.__note_mappings[current_note[0]] > 0) or (
                        12 - abs(self.__note_mappings[elem[0]] - self.__note_mappings[current_note[0]]) < min_int and
                        self.__note_mappings[
                            elem[0]] - self.__note_mappings[current_note[0]] < 0):
                    next_note = elem
                    min_int = abs(self.__note_mappings[elem[0]] - self.__note_mappings[current_note[0]])
            ordered_scale.append(next_note)
            current_note = next_note

        print(ordered_scale)
        return ordered_scale

    def pentatonic_scale_finder(self, sorted_items):
        print(sorted_items)
        note_list = sorted_items[:5]  # notes in a list
        print(note_list)
        current_note = note_list[0]
        ordered_scale = [current_note]
        while len(ordered_scale) < 5:
            min_int = 4
            next_note = current_note
            for elem in note_list:
                if (abs(self.__note_mappings[elem[0]] - self.__note_mappings[current_note[0]]) < min_int and
                    self.__note_mappings[elem[0]] -
                    self.__note_mappings[current_note[0]] > 0) or (
                        12 - abs(self.__note_mappings[elem[0]] - self.__note_mappings[current_note[0]]) < min_int and
                        self.__note_mappings[
                            elem[0]] - self.__note_mappings[current_note[0]] < 0):
                    next_note = elem
                    min_int = abs(self.__note_mappings[elem[0]] - self.__note_mappings[current_note[0]])
            ordered_scale.append(next_note)
            current_note = next_note

        print(ordered_scale)
        return ordered_scale
