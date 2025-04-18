import numpy as np
import csv
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserIconView
from kivy_garden.graph import Graph, MeshLinePlot


class MainScreen(BoxLayout):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(orientation='vertical', **kwargs)

        self.file_paths = []

        self.label = Label(text='CSV 파일 2개를 선택해주세요.')
        self.add_widget(self.label)

        self.choose_button = Button(text='CSV 파일 선택')
        self.choose_button.bind(on_press=self.select_csv_files)
        self.add_widget(self.choose_button)

        self.graph = Graph(
            xlabel='Frequency (Hz)',
            ylabel='Amplitude',
            x_ticks_minor=5,
            x_ticks_major=20,
            y_ticks_major=10,
            y_grid_label=True,
            x_grid_label=True,
            padding=5,
            x_grid=True,
            y_grid=True,
            xmin=0,
            xmax=100,
            ymin=0,
            ymax=100
        )
        self.add_widget(self.graph)

    def select_csv_files(self, instance):
        content = FileChooserIconView(filters=["*.csv"])
        popup = Popup(title="CSV 파일 선택 (2개)", content=content, size_hint=(0.9, 0.9))

        def on_selection(*args):
            if len(content.selection) == 2:
                self.file_paths = content.selection
                popup.dismiss()
                self.label.text = "선택된 파일:\n1. {}\n2. {}".format(
                    self.file_paths[0], self.file_paths[1]
                )
                self.process_csv_files()

        content.bind(on_submit=on_selection)
        popup.open()

    def process_csv_files(self):
        fft_results = []
        for path in self.file_paths:
            time, data = self.read_csv(path)
            freq, amplitude = self.compute_fft(time, data)
            fft_results.append((freq, amplitude))

        # FFT 차이 계산
        freq = fft_results[0][0]
        amp_diff = np.abs(fft_results[0][1] - fft_results[1][1])

        # 그래프 그리기
        self.graph.xmax = max(freq)
        self.graph.ymax = max(
            np.max(fft_results[0][1]),
            np.max(fft_results[1][1]),
            np.max(amp_diff)
        ) * 1.1

        self.graph.plots.clear()

        plot1 = MeshLinePlot(color=[1, 0, 0, 1])  # 빨간색
        plot1.points = list(zip(freq, fft_results[0][1]))
        self.graph.add_plot(plot1)

        plot2 = MeshLinePlot(color=[0, 1, 0, 1])  # 초록색
        plot2.points = list(zip(freq, fft_results[1][1]))
        self.graph.add_plot(plot2)

        plot_diff = MeshLinePlot(color=[1, 1, 1, 1])  # 흰색
        plot_diff.points = list(zip(freq, amp_diff))
        self.graph.add_plot(plot_diff)

    def read_csv(self, file_path):
        time = []
        data = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 헤더 건너뜀
            for row in reader:
                if len(row) >= 2:
                    try:
                        time.append(float(row[0]))
                        data.append(float(row[1]))
                    except ValueError:
                        continue
        return np.array(time), np.array(data)

    def compute_fft(self, time, data):
        n = len(data)
        dt = (time[-1] - time[0]) / n
        freq = np.fft.fftfreq(n, d=dt)[:n // 2]
        fft_vals = np.fft.fft(data)
        amplitude = np.abs(fft_vals[:n // 2])
        return freq, amplitude


class FFTApp(App):
    def build(self):
        return MainScreen()


if __name__ == '__main__':
    FFTApp().run()
