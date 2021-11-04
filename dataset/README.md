# Folder to store dataset

## Format

**images/** : store all the images

**labels/** : store all the labels, labels format: class_id, cx, cy, w, h. The last four elements are scaled to [0, 1].

**train.csv** : 

| image             | text              |
| ---------------   |:-----------------:|
| 000001.jpg        |   000001.txt      |
| 0000020.jpg        |   0000020.txt      |
|...                |   ...             |

**test.csv**

| image             | text              |
| ---------------   |:-----------------:|
| 000002.jpg        |   000002.txt      |
| 000003.jpg        |   000003.txt      |
|...                |   ...             |