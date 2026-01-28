/**
 * API Configuration
 */

// Determine API base URL
// If served from the backend port (8001), use relative paths.
// If served from file://, default to localhost:8001.
// Otherwise (e.g., served from 8000), point to same hostname but port 8001.
const BACKEND_PORT = '8001';
let apiBase = `http://localhost:${BACKEND_PORT}`;

// If we are running locally (localhost or file), we might want to use the current protocol/host
// But usually, the backend is always HTTP localhost:8001 unless we are proxying.
// If served from the backend (localhost:8001), use relative path or same origin.
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    // If we are on the backend port, use same origin
    if (window.location.port === BACKEND_PORT) {
        apiBase = `${window.location.protocol}//${window.location.hostname}:${BACKEND_PORT}`;
    }
    // If we are on a different port (e.g. 5500 via Live Server), default to localhost:8001
} else {
    // If we are on GitHub Pages or another remote host, we MUST point to localhost:8001
    // Note: Mixed Content (HTTPS -> HTTP) might be an issue on GitHub Pages.
    apiBase = `http://localhost:${BACKEND_PORT}`;
}

const API_BASE = apiBase;

export { API_BASE, BACKEND_PORT };
