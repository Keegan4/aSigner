FROM node:24-slim
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Rewrites are resolved at build time, so the API address must be known here.
ARG API_URL=http://server:8000
ENV API_URL=$API_URL
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "start"]
