FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "week4_homework.py", "./" ]

ENTRYPOINT [ "python", "week4_homework.py", "yellow", "2023", "4"]
