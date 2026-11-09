FROM node:20

WORKDIR /app

COPY package.json package-lock.json* ./

RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]

# To build the docker image run (type the --build on first run or when adding a new package):
    # docker compose up --build