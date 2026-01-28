/**
 * AI Concepts Demo - Main Entry Point
 */

import { checkHealth } from './api.js';
import { initTabs, initSubTabs } from './tabs.js';
import { initLLM } from './llm.js';
import { initEmbeddings } from './embeddings.js';
import { initRAG } from './rag.js';
import { initAgent } from './agent.js';
import { initSettings } from './settings.js';

/**
 * Initialize the setup banner close button
 */
function initBanner() {
    const closeBtn = document.getElementById('banner-close-btn');
    const banner = document.getElementById('setup-banner');

    if (closeBtn && banner) {
        closeBtn.addEventListener('click', () => {
            banner.style.display = 'none';
        });
    }
}

/**
 * Initialize the application
 */
document.addEventListener('DOMContentLoaded', () => {
    initBanner();
    initTabs();
    initSubTabs();
    initLLM();
    initEmbeddings();
    initRAG();
    initAgent();
    initSettings();

    // Check health on load and periodically
    checkHealth();
    setInterval(checkHealth, 30000);
});
