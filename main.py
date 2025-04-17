import numpy as np
from numpy.fft import fft, fftfreq
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy_garden.graph import Graph, MeshLinePlot
from plyer import filechooser
from kivy.utils import platform
import csv

def smooth_data(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def compute_fft(time_data, acc_data):
    n = len(acc_data)
    if n < 2:
        return [], 1, 1
    timestep = (time_data[-1] - time_data[0]) / n
    freq = fftfreq(n, d=timestep)[:n // 2]
    fft_values = np.abs(fft(acc_data))[:n // 2]
    valid = freq <= 50
    smoothed = smooth_data(fft_values[valid])
    return list(zip(freq[valid], smoothed)), max(freq[valid]), max(smoothed)

class GraphWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.graph = Graph(
            xlabel='Frequency (Hz)', ylabel='Amplitude',
            x_ticks_major=10, y_ticks_major=1,
            x_grid=True, y_grid=True,
            xmin=0, xmax=50, ymin=0, ymax=1
        )
        self.add_widget(self.graph)
        self.plots = []

    def update_graph(self, fft_points, x_max, y_max):
        self.graph.xmax = max(x_max, 1)
        self.graph.ymax = max(y_max, 0.1)
        for plot in self.plots:
            self.graph.remove_plot(plot)
        self.plots.clear()
        plot = MeshLinePlot(color=[0, 1, 0, 1])
        plot.points = fft_points
        self.graph.add_plot(plot)
        self.plots.append(plot)

class CSVFFTApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical', spacing=5, padding=5)

        self.select_button = Button(text="Select CSV", size_hint=(1, 0.1))
        self.select_button.bind(on_press=self.select_csv_file)
        self.layout.add_widget(self.select_button)

        self.graph_widget = GraphWidget(size_hint=(1, 0.7))
        self.layout.add_widget(self.graph_widget)

        self.console = TextInput(readonly=True, multiline=True, size_hint=(1, 0.2),
                                 background_color=[0, 0, 0, 1], foreground_color=[0, 1, 0, 1])
        self.layout.add_widget(self.console)

        return self.layout

    def log(self, msg):
        print(msg)
        self.console.text += msg + "\n"

    def select_csv_file(self, instance):
        def on_selection(selection):
            if not selection:
                self.log("파일이 선택되지 않았습니다.")
                return
            self.process_csv(selection[0])

        if platform == 'android':
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.READ_EXTERNAL_STORAGE])
        filechooser.open_file(on_selection=on_selection)

    def process_csv(self, file_path):
        try:
            time_data = []
            acc_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    try:
                        time_data.append(float(row[0]))
                        acc_data.append(float(row[1]))
                    except:
                        continue
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='cp949') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    try:
                        time_data.append(float(row[0]))
                        acc_data.append(float(row[1]))
                    except:
                        continue

        fft_points, x_max, y_max = compute_fft(time_data, acc_data)
        self.log(f"{file_path} 처리 완료: {len(fft_points)}개 포인트")
        Clock.schedule_once(lambda dt: self.graph_widget.update_graph(fft_points, x_max, y_max))

if __name__ == "__main__":
    CSVFFTApp().run()
