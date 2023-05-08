FROM python:3.9

WORKDIR /app

COPY ./requirements.txt ./

RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./deploy ./deploy
COPY ./src/data ./src/data
COPY ./configs ./configs
COPY ./src/fusion/fusion_thresholds_mel_modgd_pitch_acc.npy ../src/fusion/fusion_thresholds_mel_modgd_pitch_acc.npy
COPY ./models ../models

# moze biti i obrisano
ENV LISTEN_PORT=5050
EXPOSE 5050

CMD ["python3", "-m", "deploy.rest_api"]
# ENTRYPOINT ["python3", "-m", "deploy.rest_api"]