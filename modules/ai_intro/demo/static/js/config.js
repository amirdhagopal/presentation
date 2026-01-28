/**
 * API Configuration
 */

// Determine API base URL
// If served from the backend port (8001), use relative paths.
// If served from file://, default to localhost:8001.
// Otherwise (e.g., served from 8000), point to same hostname but port 8001.
const BACKEND_PORT = '8001';
let apiBase = '';

if (window.location.protocol === 'file:') {
    apiBase = `http://localhost:${BACKEND_PORT}`;
} else if (window.location.port !== BACKEND_PORT) {
    apiBase = `${window.location.protocol}//${window.location.hostname}:${BACKEND_PORT}`;
}

const API_BASE = apiBase;

export { API_BASE, BACKEND_PORT };
