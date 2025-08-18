## e-cup-ozon ML framework


Минимальный каркас для соревнований: единый API моделей, тайм-сплиты и фичи, автоматизированные train/validate/infer, логирование в MLflow, пайплайн через DVC, конфигурация через Hydra.

### 1) Требования
- Python 3.10–3.11
- Poetry ≥ 2.0 (для зависимостей)
- Учетная запись Hugging Face (для доступа к `hf://datasets/...`)

### 2) Установка окружения
```bash
cd /Users/m.s.sobolev/e-cup-ozon
# Рекомендуем локальный venv проекта
poetry config virtualenvs.in-project true --local
poetry env use python3.10  # или ваш python3
poetry install             # базовые зависимости
# доп. зависимости для RecBole (опционально)
poetry install --with recbole
```
Авторизация в Hugging Face:
```bash
huggingface-cli login
```

macOS + LightGBM (если используете LGBM):
```bash
brew install libomp
```

### 3) MLflow
По умолчанию используется локальный трекинг (`./mlflow`, смотрите `conf/config.yaml`). Запуск UI:
```bash
mlflow ui --backend-store-uri ./mlflow
```
Для удалённого трекинга замените `mlflow.tracking_uri` в `conf/config.yaml` и/или экспортируйте переменные окружения (`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`).

### 4) DVC
Инициализация (если ещё не выполнено):
```bash
git init
poetry run dvc init
git add .
git commit -m "Init Git + DVC"
```
(Опционально) подключите удалённый remote (`dvc remote add -d ...`) и затем `dvc push`.

### 5) Структура проекта (ключевое)
```
conf/                 # конфиги Hydra
  config.yaml         # дефолтная сборка
  data/flat50m.yaml   # источники/пути данных
  split/time.yaml     # параметры тайм-сплитов
  model/{dummy,lgbm,recbole}.yaml
  pipeline/{train,infer}.yaml
ml/
  core/               # data/splits/features, evaluator, utils
  models/             # адаптеры моделей (Dummy, LGBM, RecBole)
  prepare_data.py     # prepare stage
  make_split.py       # split stage
  build_features.py   # features stage
  train.py            # обучение + валидация + сохранение модели
  infer.py            # инференс из сохранённой модели
artifacts/            # артефакты (модель, метрики, предсказания)
data/                 # данные (под управлением DVC)
dvc.yaml              # декларация пайплайна
```

### 6) Запуск пайплайна DVC
Полный прогон всех этапов:
```bash
poetry run dvc repro
```
Запуск с конкретного этапа:
```bash
# только features
poetry run dvc repro features
# с этапа и вниз по графу
poetry run dvc repro --downstream split
# форс-перезапуск этапа
poetry run dvc repro -f train
```

### 7) Обучение (в обход DVC)
- По умолчанию модель выбирается через Hydra-группу `model`. Доступные конфиги: `conf/model/*.yaml`.
```bash
# Dummy
poetry run python -m ml.train model=dummy
# LightGBM (бинарная классификация, metric=AUC)
poetry run python -m ml.train model=lgbm
# RecBole (опционально, нужен poetry install --with recbole)
poetry run python -m ml.train model=recbole model.params.model=BPR
```
Переопределение гиперпараметров на лету:
```bash
poetry run python -m ml.train model=lgbm \
  model.params.num_boost_round=500 model.params.learning_rate=0.03
```
Артефакты:
- Модель: `artifacts/model/` (`model.txt` для LGBM, `user_mean.parquet` для Dummy)
- Метрики: `artifacts/metrics.json`

### 8) Инференс
Модель должна быть сохранена в `artifacts/model/` предыдущим шагом обучения.
```bash
# по умолчанию split=test, не передавайте split=test (Hydra воспримет как выбор группы)
poetry run python -m ml.infer
# если нужен другой сплит — сначала убедитесь, что он существует в data/splits/*.parquet
```
Если видите ошибку вида `Could not find 'split/test'` — не передавайте `split=test` как оверрайд Hydra. В `dvc.yaml` также рекомендуется убрать `split=test` из команды инференса (используйте просто `python -m ml.infer`).

### 9) Версионирование данных и артефактов
После прогонов:
```bash
poetry run dvc add data/splits data/features artifacts/model artifacts/metrics.json
git add data/.gitignore artifacts/.gitignore *.dvc
git commit -m "Versioned data, features, and model"
# при наличии удалённого DVC-remote
poetry run dvc push
```

### 10) CI (GitHub Actions)
- Файл: `.github/workflows/ci.yaml`
- Действия: checkout → Poetry install → запуск тренировки (`poetry run python -m ml.train model.name=dummy`) → выгрузка артефактов.
- Для RecBole добавьте установку группы: `poetry install --with recbole` и поменяйте команду тренировки.

### 11) Настройка модели
- Через Hydra: `model=<name>` выбирает конфиг `conf/model/<name>.yaml`.
- В коде `ml/train.py` модель создаётся через `resolve_model_from_cfg` по `cfg.model.name` и `cfg.model.params`.
- Для LGBM исключены признаки с утечкой (`played_ratio_pct`). Конфиг по умолчанию: бинарный objective + AUC.

### 12) Частые проблемы и решения
- macOS + LightGBM: `libomp.dylib not found`
  ```bash
  brew install libomp
  # при необходимости
  export DYLD_LIBRARY_PATH="$(brew --prefix libomp)/lib:$DYLD_LIBRARY_PATH"
  ```
- Запускается python 2.7 (SyntaxError на аннотациях):
  - Используйте `poetry run ...` или явный `python3`.
- Hydra конфликт для `split=test`:
  - Не передавайте `split=test`. Используйте дефолт (`test`) или измените код/команду на свой параметр (например, хранить сплит в другом ключе и читать его в `ml/infer.py`).
- `FileNotFoundError: data/splits/test.parquet`:
  - Сначала создайте сплиты: `poetry run python -m ml.make_split` или `poetry run dvc repro split`.
- Нет артефактов модели при инференсе:
  - Сначала обучите модель: `poetry run python -m ml.train model=lgbm` (или `model=dummy`).

### 13) Быстрый старт (шаги)
```bash
# 1. Установка
poetry config virtualenvs.in-project true --local
poetry env use python3.10
poetry install
huggingface-cli login

# 2. Полный пайплайн
poetry run dvc repro

# 3. Обучение вручную
poetry run python -m ml.train model=lgbm

# 4. Инференс (по умолчанию test)
poetry run python -m ml.infer

# 5. Версионирование и публикация
poetry run dvc add data/splits data/features artifacts/model artifacts/metrics.json
poetry run dvc push
```
