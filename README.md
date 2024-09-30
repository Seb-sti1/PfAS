# 34759 - Perception for Autonomous Systems

## Get started

```sh
docker build --tag=sebsti1/pfas . # build image
# **OR**
docker pull sebsti1/pfas
# **THEN**
docker run -it -p 127.0.0.1:8080:8080 -v ./src:/root/src --rm sebsti1/pfas /bin/bash # start image
```

Then go to the url provided in the last output of the command
(e.g. `http://127.0.0.1:8080/lab?token=ba2ae728b9007cea2305540b8680fb492966f9f5de1dc5b4`).
(e.g. `http://127.0.0.1:8080/lab?token=ba2ae728b9007cea2305540b8680fb492966f9f5de1dc5b4`).
All the python files will be created in the `src/` folder.

## Known issues

- Unless used with some [X11 support](https://github.com/Seb-sti1/robot_autonomy/blob/master/docker-compose.yml#L6)
  the open-cv and open3d window won't open
- open3d transform function won't work with the last version of numpy, use numpy==1.26.4

## LICENSE

No license is provided because certain files come from the course and I only have the right to use them. 