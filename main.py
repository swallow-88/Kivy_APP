
# -*- coding: utf-8 -*-
import os
import numpy as np
import csv
from numpy.fft import fft, fftfreq
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.utils import platform
from kivy_garden.graph import Graph, MeshLinePlot
from plyer import filechooser
from android.permissions import request_permissions, Permission
from android import mActivity
from jnius import autoclass, cast

def check_storage_permission():
    Environment = autoclass('android.os.Environment')
    if not Environment.getExternalStorageState() == Environment.MEDIA_MOUNTED:
        raise PermissionError("외부 저장소에 접근할 수 없습니다.")

def request_android_permissions():
    request_permissions([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE])

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
            [1, 0, 0, 1],  # CSV 1 - ë¹¨ê°ì
            [0, 1, 0, 1],  # CSV 2 - ë¹ì
        ]

        for i, points in enumerate(points_list):
            color = fixed_colors[i] if i < len(fixed_colors) else [0.5, 0.5, 0.5, 1]
            plot = MeshLinePlot(color=color)
            plot.points = points
            self.graph.add_plot(plot)
            self.plots.append(plot)

        if diff_points:
            diff_plot = MeshLinePlot(color=[1, 1, 1, 1])  # í°ì
            diff_plot.points = diff_points
            self.graph.add_plot(diff_plot)
            self.plots.append(diff_plot)

class FFTApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', spacing=5, padding=5)

        self.graph_widget = GraphWidget(size_hint=(1, 0.7))
        self.layout.add_widget(self.graph_widget)

        self.console = TextInput(readonly=True, multiline=True, size_hint=(1, 0.2),
                                background_color=[0, 0, 0, 1], foreground_color=[0, 1, 0, 1])
        self.layout.add_widget(self.console)

        self.load_button = Button(text="CSV íì¼ ì í ë° ë¶ì", size_hint=(1, 0.1))
        self.load_button.bind(on_press=self.select_csv_files)
        self.layout.add_widget(self.load_button)

        return self.layout

    def log(self, msg):
        print(msg)
        self.console.text += msg + "\n"

    def show_popup(self, message):
        popup = Popup(title='ìë¦¼', content=Label(text=message), size_hint=(0.8, 0.3))
        popup.open()

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
        except FileNotFoundError:
            self.log("파일을 찾을 수 없습니다.")
            return [], 1, 1
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='cp949') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        try:
                            time_vals.append(float(row[0]))
                            acc_vals.append(float(row[1]))
                        except (ValueError, IndexError):
                            continue
            except Exception as e:
                self.log(f"파일 처리 중 오류 발생: {e}")
                return [], 1, 1
        except Exception as e:
            self.log(f"파일 처리 중 오류 발생: {e}")
            return [], 1, 1

        fft_data, x_max, y_max = compute_fft(time_vals, acc_vals)
        self.log(f"{file_path} 처리 완료: {len(fft_data)}개 포인트")
        return fft_data, x_max, y_max

    def select_csv_files(self, instance):
        if platform != 'android':
            self.show_popup("ì´ ê¸°ë¥ì Android ê¸°ê¸°ììë§ ìëí©ëë¤.")
            return

        def on_selection(selection):
            if not selection or len(selection) < 2:
                self.show_popup("CSV íì¼ 2ê°ë¥¼ ì íí´ì£¼ì¸ì.")
                return

            file1, file2 = selection[:2]
            if not all(f.endswith('.csv') and os.path.exists(f) for f in [file1, file2]):
                self.show_popup("CSV íì¼ì´ ìëê±°ë ì¡´ì¬íì§ ììµëë¤.")
                return

            fft1, x1, y1 = self.process_csv(file1)
            fft2, x2, y2 = self.process_csv(file2)
            diff = compute_fft_difference(fft1, fft2)
            x_max = max(x1, x2)
            y_max = max(y1, y2, max(y for _, y in diff))
            Clock.schedule_once(lambda dt: self.graph_widget.update_graph([fft1, fft2], diff, x_max, y_max))

        filechooser.open_file(on_selection=on_selection, multiple=True)

if __name__ == "__main__":
    FFTApp().run()
