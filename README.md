# 34759 - Perception for Autonomous Systems


## Get started

```sh
docker build --tag=sebsti1/pfas . # build image
docker run -it -p 127.0.0.1:8080:8080 -v ./src:/root/src --rm sebsti1/pfas /bin/bash # start image
```

Then go to the url provided in the output of the command (e.g. `http://127.0.0.1:8080/lab?token=ba2ae728b9007cea2305540b8680fb492966f9f5de1dc5b4`).
All the python files will be create in the `src/` folder.
