# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 08:43:22 2025

@author: 6402078
"""

import numpy as np
from numpy.fft import fft, fftfreq
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.clock import Clock
from kivy.uix.screenmanager import Screen
from kivy_garden.graph import Graph, MeshLinePlot
from plyer import accelerometer, filechooser
from kivy.utils import platform
from kivy.uix.textinput import TextInput
import time
import csv
import threading

# ----------------------- FFT 분석 함수 -----------------------
def smooth_data(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def compute_fft(time_data, acceleration_data):
    n = len(acceleration_data)
    if n < 2:
        return [], 1, 1
    timestep = (time_data[-1] - time_data[0]) / n
    freq = fftfreq(n, d=timestep)[:n // 2]
    fft_values = np.abs(fft(acceleration_data))[:n // 2]
    valid_indices = freq <= 50
    smoothed = smooth_data(fft_values[valid_indices])
    x_max = max(freq[valid_indices])
    y_max = max(smoothed)
    return list(zip(freq[valid_indices], smoothed)), x_max, y_max

def compute_fft_difference(fft1, fft2):
    freq1, mag1 = zip(*fft1)
    freq2, mag2 = zip(*fft2)
    common_freq = np.linspace(0, 50, 200)
    mag1_interp = np.interp(common_freq, freq1, mag1)
    mag2_interp = np.interp(common_freq, freq2, mag2)
    diff = np.abs(mag1_interp - mag2_interp)
    return list(zip(common_freq, diff))

# ----------------------- 메인 화면 -----------------------
def load_csv_file(file_path):
    try:
        print(f"불러온 파일 경로: {file_path}")
        if not file_path or not file_path.endswith('.csv'):
            raise ValueError("CSV 파일이 아닙니다.")
        time_data = []
        acc_data = []

        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 헤더 건너뛰기
            for row in reader:
                try:
                    time_data.append(float(row[0]))
                    acc_data.append(float(row[1]))
                except Exception as e:
                    print(f"데이터 파싱 오류: {e}")
                    continue

        if len(time_data) == 0 or len(acc_data) == 0:
            raise ValueError("CSV에 유효한 데이터가 없습니다.")

        return time_data, acc_data

    except Exception as e:
        print(f"[오류] CSV 파일 불러오기 실패: {e}")
        return None, None

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.acceleration_data = []

    def enable_accelerometer(self):
        try:
            accelerometer.enable()
            print('Accelerometer enabled successfully')
        except NotImplementedError:
            print("Don't use the Accelerometer")

# ----------------------- 그래프 위젯 -----------------------
class GraphWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.plots = []

        self.graph = Graph(
            xlabel='Frequency (Hz)',
            ylabel='Amplitude',
            x_ticks_minor=5,
            x_ticks_major=10,
            y_ticks_major=1,
            y_grid_label=True,
            x_grid_label=True,
            padding=5,
            xlog=False,
            ylog=False,
            x_grid=True,
            y_grid=True,
            xmin=0,
            xmax=50,
            ymin=0,
            ymax=1
        )
        self.add_widget(self.graph)

    def update_graph(self, points_list, diff_points, x_max, y_max):
        self.graph.xmax = max(float(x_max), 1)
        self.graph.ymax = max(float(y_max), 0.1)

        for plot in self.plots:
            self.graph.remove_plot(plot)
        self.plots.clear()

        fixed_colors = [
            [0, 1, 0, 1],  # CSV 1 - 녹색
            [1, 0, 0, 1],  # CSV 2 - 빨강
            [1, 0, 0, 1],  # X - 빨강
            [0, 0, 1, 1],  # Y - 파랑
            [0, 1, 0, 1]   # Z - 초록
        ]

        for i, points in enumerate(points_list):
            color = fixed_colors[i] if i < len(fixed_colors) else [0.5, 0.5, 0.5, 1]
            plot = MeshLinePlot(color=color)
            plot.points = points
            self.graph.add_plot(plot)
            self.plots.append(plot)

        if diff_points:
            plot = MeshLinePlot(color=[1, 1, 1, 1])
            plot.points = diff_points
            self.graph.add_plot(plot)
            self.plots.append(plot)

# ----------------------- 메인 앱 -----------------------
class FFTApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', spacing=5, padding=5)

        self.spinner = Spinner(
            text='CSV to JSON',
            values=('CSV to JSON', 'Real-time Accelerometer', 'Real-time Microphone'),
            size_hint=(1, 0.1)
        )
        self.layout.add_widget(self.spinner)

        # 처리 버튼
        self.process_button = Button(text="Process Data", size_hint=(1, 0.1))
        self.process_button.bind(on_press=self.process_data)
        self.layout.add_widget(self.process_button)

        # 그래프 위젯
        self.graph_widget = GraphWidget(size_hint=(1, 0.6))
        self.layout.add_widget(self.graph_widget)

        # 그래프 리셋 버튼
        self.reset_button = Button(text="Reset Graph", size_hint=(1, 0.1))
        self.reset_button.bind(on_press=self.reset_graph)
        self.layout.add_widget(self.reset_button)

        # 콘솔 로그창
        self.console = TextInput(readonly=True, multiline=True, size_hint=(1, 0.2),
                                 background_color=[0, 0, 0, 1], foreground_color=[0, 1, 0, 1])
        self.layout.add_widget(self.console)

        self.main_screen = MainScreen()
        return self.layout

    def log(self, message):
        print(message)
        self.console.text += message + "\n"

    def on_start(self):
        self.main_screen.enable_accelerometer()
        self.log("Start APP - Activation Sensor")

    def reset_graph(self, instance):
        for plot in self.graph_widget.plots:
            self.graph_widget.graph.remove_plot(plot)
        self.graph_widget.plots.clear()
        self.log("Complete Graph reset")

    def process_data(self, instance):
        if self.spinner.text == 'CSV to JSON':
            file1 = "/sdcard/Download/file1.csv"
            file2 = "/sdcard/Download/file2.csv"
            self.log(f'직접 경로로 처리: {file1}, {file2}')
            fft1, x1, y1 = self.process_csv(file1)
            fft2, x2, y2 = self.process_csv(file2)
            diff = compute_fft_difference(fft1, fft2)
            x_max = max(x1, x2)
            y_max = max(y1, y2, max(y for _, y in diff))
            Clock.schedule_once(lambda dt: self.graph_widget.update_graph([fft1, fft2], diff, x_max, y_max))
            # 테스트용 FFT 시각화 (선택)
            '''
            t = np.linspace(0, 10, 1000)
            a = 0.5 * np.sin(2 * np.pi * 5 * t)
            fft_result, x_max, y_max = compute_fft(t, a)
            Clock.schedule_once(lambda dt: self.graph_widget.update_graph([fft_result], [], x_max, y_max))
            self.log("테스트용 FFT 그래프 출력")
            '''
            #self.select_csv_files()  # 실제 파일 선택으로 전환하려면 이 줄을 주석 해제
        elif self.spinner.text == 'Real-time Accelerometer':
            threading.Thread(target=self.collect_accelerometer_data).start()
        elif self.spinner.text == 'Real-time Microphone':
            threading.Thread(target=self.simulate_microphone_data).start()

    def select_csv_files(self):
        self.log("Called filechooser")
        def on_selection(selection):
            self.log(f"on_selection callback: {selection}")
            if not selection or len(selection) < 2:
                self.log("Select 2 CSV files.")
                return

            selected_files = selection[:2]
            self.log(f"Selected files: {selected_files}")

            try:
                fft1, x1, y1 = self.process_csv(selected_files[0])
                fft2, x2, y2 = self.process_csv(selected_files[1])

                if not fft1 or not fft2:
                    self.log("Don't have the FFT result")
                    return

                diff = compute_fft_difference(fft1, fft2)
                x_max = max(x1, x2)
                y_max = max(y1, y2, max(y for _, y in diff))

                Clock.schedule_once(lambda dt: self.graph_widget.update_graph(
                    [fft1, fft2], diff, x_max, y_max))
                self.log("Complete Graph update.")
            except Exception as e:
                self.log(f"FFT error: {str(e)}")

        if platform in ('android', 'ios'):
            filechooser.open_file(on_selection=on_selection, multiple=True)
        else:
            self.log("Only select file in Android")

    def process_csv(self, file_path):
        time_vals = []
        acc_vals = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    try:
                        time_vals.append(float(row[0]))
                        acc_vals.append(float(row[1]))
                    except (ValueError, IndexError):
                        continue
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='cp949') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    try:
                        time_vals.append(float(row[0]))
                        acc_vals.append(float(row[1]))
                    except (ValueError, IndexError):
                        continue

        fft_data, x_max, y_max = compute_fft(time_vals, acc_vals)
        self.log(f"{file_path} 처리 완료: {len(fft_data)}개 포인트")
        return fft_data, x_max, y_max

    def collect_accelerometer_data(self, duration=10):
        try:
            accelerometer.enable()
        except NotImplementedError:
            self.log("Don't use the sensor")
            return

        self.accel_data = []
        start_time = time.time()

        while time.time() - start_time < duration:
            acc = accelerometer.acceleration
            timestamp = time.time() - start_time
            if acc != (None, None, None):
                self.accel_data.append((timestamp, acc[0], acc[1], acc[2]))
            time.sleep(0.01)

        self.log("Complete ACC data")
        self.process_accel_data_xyz()

    def process_accel_data_xyz(self):
        data = np.array(self.accel_data)
        time_vals = data[:, 0]
        x_vals = data[:, 1]
        y_vals = data[:, 2]
        z_vals = data[:, 3]

        fft_x, x1, y1 = compute_fft(time_vals, x_vals)
        fft_y, x2, y2 = compute_fft(time_vals, y_vals)
        fft_z, x3, y3 = compute_fft(time_vals, z_vals)

        x_max = max(x1, x2, x3)
        y_max = max(y1, y2, y3)

        Clock.schedule_once(lambda dt: self.graph_widget.update_graph(
            [fft_x, fft_y, fft_z], [], x_max, y_max))
        self.log("Real time ACC and FFT graph")

    def simulate_microphone_data(self):
        t = np.linspace(0, 10, 1000)
        mic = 0.7 * np.sin(2 * np.pi * 10 * t)
        fft_result, x_max, y_max = compute_fft(t, mic)
        Clock.schedule_once(lambda dt: self.graph_widget.update_graph([fft_result], [], x_max, y_max))
        self.log("Real time mike and FFT graph")
        
        
# ----------------------- 실행 -----------------------
if __name__ == "__main__":
    FFTApp().run()
