import pandas as pd

dataset = "ucf-crime"
padding = True

data_file_train = f"data/{dataset}/{dataset}.training.csv"  # video list


video_list = pd.read_csv(data_file_train)
video_list = video_list["video-id"].values[:]

anomaly = video_list[:6614]
regular = video_list[6614:]

anomaly_list_first = []
anomaly_list_last = []
for x in anomaly:
    if int(x.split("-")[1]) % 4 == 3:
        anomaly_list_last.append(x)
    else:
        anomaly_list_first.append(x)

regular_list_first = []
regular_list_last = []
for x in regular:
    if int(x.split("-")[1]) % 4 == 3:
        regular_list_last.append(x)
    else:
        regular_list_first.append(x)

print(len(anomaly_list_first), len(regular_list_first))
print(len(anomaly_list_last), len(regular_list_last))

anomaly_list = anomaly_list_first + anomaly_list_last
regular_list = regular_list_first + regular_list_last
file_ids = anomaly_list + regular_list
df = pd.DataFrame(file_ids, columns=["video-id"])
df.to_csv(f"data/{dataset}/{dataset}-ensemble-split-p.training.csv")
