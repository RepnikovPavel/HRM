# metrics-viz: exact accuracy(t) для прогонов судоку

Самописный мини-tensorboard с микрофронтенд-архитектурой виджетов (как
tviz: `~/calc/modeling/tviz/front/`): shell с перетаскиваемыми и
ресайзабельными тайлами (drag за заголовок, resize за правый нижний угол,
закрытие, добавление через «+ виджет»), layout и состояние виджетов
(таймфрейм, выбранная модель, рисунки) — в localStorage браузера.

Структура:

- `viz/front/index.html` — топбар, доска тайлов, стили (css-переменные
  как у tviz).
- `viz/front/shell/shell.js` — тайлы, picker виджетов, загрузка
  `/api/series` каждые 60 с, шина `onData` для виджетов.
- `viz/front/shell/registry.js` — реестр виджетов.
- `viz/front/widgets/lines.js` — exact accuracy обеих моделей; селектор
  режима отрисовки: линия, линия с точками, маркеры измерений,
  прерывистая, прерывистая с точками, ступени до/по центру/после
  (по умолчанию «ступени (после)»: значение метрики держится до следующего
  замера); drag — панорама, колесо — зум, кроссхеар.
- `viz/front/widgets/candles.js` — полный порт tviz
  `front/widgets/candles.js` на метрики: выбор модели (select), таймфрейм
  15м/30м/1ч/2ч/4ч (клавиши 1-5), свеча = OHLC exact accuracy точек внутри
  бакета, тела всегда закрашены; магнит (snap 8px к OHLC), go-live ►|
  (followTail за новыми данными), y-ось drag/dblclick, кроссхеар с
  OHLC-легендой внутри графика слева вверху; инструменты: курсор, линейка
  (в т.ч. shift+2 клика), карандаш, тренд-линия, горизонталь, фибоначчи
  (0/23.6/38.2/50/61.8/78.6/100% с подписями); индикаторы fx: SMA(20),
  EMA(50), Bollinger(20,2), RSI(14) и MACD(12,26,9) в суб-пейнах.
  Объёмов нет (метрика, не цена): без суб-пейна объёма, VP и VWAP.
  Рисунки персистятся в localStorage раздельно по моделям
  (drawingsByRun — при переключении модели чужие уровни не тянутся).
  Панорама drag по обеим осям (вертикаль сдвигает yRange через yShift,
  первый вертикальный drag выключает y-автомасштаб; dblclick по оси цены
  сбрасывает масштаб и сдвиг).

Сервер: `viz/metrics_server.py` (stdlib http.server) отдаёт статику
`viz/front/` и `/api/series` (JSON из metrics.jsonl прогонов на лету:
x — часы с начала прогона, y — exact %).

Запуск на сервере:

    ssh user@192.168.0.1 'cd ~/HRM && DATA_ROOT=/mnt/hdd2/hrm bash scripts/serve_metrics_viz.sh'

Постоянный доступ из LAN без туннеля (docker публикует 0.0.0.0:8377):

    http://192.168.0.1:8377

Серии: HRM 27M (train exact на train-батчах, sudoku-1k-server-v2) и
RRN 27M (test exact на 8192, rrn27m-sudoku-1k). Образ
`hrm-metrics-viz:local` (docker/DockerfileMetricsViz, python:3.12-slim).
