# pacerr-testing-env

---
This is a testing environment for the backend architecture of the PACERR (Plant Analysis Controlled Environment Research Room) project. This specific container is proof of concept work that shows the ability to run an object recognition server using flask and pytorch.

### Building the Container
`docker build -t pacerr .`

### Running the Container
`docker run -p 5000:5000 pacerr`