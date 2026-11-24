# Stage 1: Build TypeScript source into JavaScript
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package manifests and install all dependencies (including devDependencies)
COPY package*.json ./
RUN npm install

# Copy TypeScript configuration and source code, then compile
COPY tsconfig.json ./
COPY src ./src
RUN npm run build


# Stage 2: Lightweight production runtime with FFmpeg
FROM node:22-alpine AS production

# Install FFmpeg for video frame extraction
RUN apk add --no-cache ffmpeg

WORKDIR /app

ENV NODE_ENV=production

# Copy package manifests and install production dependencies only
COPY package*.json ./
RUN npm install --omit=dev

# Copy compiled JavaScript output from the builder stage
COPY --from=builder /app/dist ./dist

# Create temporary directory for uploaded videos and extracted frames
RUN mkdir -p /app/tmp && chown -R node:node /app

USER node

EXPOSE 3005

CMD ["node", "dist/index.js"]

# TO BUILD THE DOCKER IMAGE RUN:
# docker build -t backend .

# TO RUN THE CONTAINER ON LINUX RUN:
# docker run --rm \
#   --name backend \
#   -p 3005:3005 \
#   --add-host=host.docker.internal:host-gateway \
#   -e PORT=3005 \
#   -e DETECTION_SERVICE_URL=http://host.docker.internal:8001 \
#   -e SIGNATURE_SERVICE_URL=http://host.docker.internal:8004 \
#   -e TRACKER_SERVICE_URL=http://host.docker.internal:8002 \
#   -e CONCURRENCY_LIMIT=5 \
#   backend

# TO RUN THE CONTAINER ON WINDOWS RUN:
# docker run --rm `
#   --name backend `
#   -p 3005:3005 `
#   --add-host=host.docker.internal:host-gateway `
#   -e PORT=3005 `
#   -e DETECTION_SERVICE_URL=http://host.docker.internal:8001 `
#   -e SIGNATURE_SERVICE_URL=http://host.docker.internal:8004 `
#   -e TRACKER_SERVICE_URL=http://host.docker.internal:8002 `
#   -e CONCURRENCY_LIMIT=5 `
#   backend
