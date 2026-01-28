/**
 * Settings Modal
 */

import { API_BASE } from './config.js';
import { checkHealth } from './api.js';

let availableModels = [];

async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/api/models`);
        const data = await response.json();
        availableModels = data.models || [];
        return {
            models: availableModels,
            currentLlm: data.current_llm,
            currentEmbed: data.current_embed
        };
    } catch (error) {
        console.error('Failed to load models:', error);
        return { models: [], currentLlm: '', currentEmbed: '' };
    }
}

function initSettings() {
    const settingsBtn = document.getElementById('settings-btn');
    const settingsModal = document.getElementById('settings-modal');
    const settingsClose = document.getElementById('settings-close');
    const settingsSave = document.getElementById('settings-save');
    const settingsReset = document.getElementById('settings-reset');
    const llmSelect = document.getElementById('llm-model-select');
    const embedSelect = document.getElementById('embed-model-select');
    const currentConfigEl = document.getElementById('current-config');

    async function openSettings() {
        settingsModal.classList.remove('hidden');

        const { models, currentLlm, currentEmbed } = await loadModels();

        // Populate LLM models
        const llmModels = models.filter(m => m.type === 'llm');
        llmSelect.innerHTML = llmModels.map(m =>
            `<option value="${m.name}" ${m.name === currentLlm ? 'selected' : ''}>${m.name} (${m.size_gb}GB)</option>`
        ).join('');

        // Populate embedding models
        const embedModels = models.filter(m => m.type === 'embedding');
        embedSelect.innerHTML = embedModels.map(m =>
            `<option value="${m.name}" ${m.name === currentEmbed ? 'selected' : ''}>${m.name} (${m.size_gb}GB)</option>`
        ).join('');

        // If no embedding models found, show all models
        if (embedModels.length === 0) {
            embedSelect.innerHTML = models.map(m =>
                `<option value="${m.name}" ${m.name === currentEmbed ? 'selected' : ''}>${m.name}</option>`
            ).join('');
        }

        currentConfigEl.innerHTML = `Current: <strong>${currentLlm}</strong> (LLM) | <strong>${currentEmbed}</strong> (Embed)`;
    }

    function closeSettings() {
        settingsModal.classList.add('hidden');
    }

    async function saveSettings() {
        const llmModel = llmSelect.value;
        const embedModel = embedSelect.value;

        try {
            const response = await fetch(
                `${API_BASE}/api/config?llm_model=${encodeURIComponent(llmModel)}&embed_model=${encodeURIComponent(embedModel)}`,
                { method: 'POST' }
            );
            const data = await response.json();

            // Update footer with new models
            updateFooterModels(data.llm_model, data.embed_model);

            closeSettings();
            checkHealth(); // Refresh status
        } catch (error) {
            console.error('Failed to save settings:', error);
            alert('Failed to save settings');
        }
    }

    async function resetSettings() {
        try {
            const configResponse = await fetch(`${API_BASE}/api/config`);
            const config = await configResponse.json();

            const response = await fetch(
                `${API_BASE}/api/config?llm_model=${encodeURIComponent(config.defaults.llm_model)}&embed_model=${encodeURIComponent(config.defaults.embed_model)}`,
                { method: 'POST' }
            );
            const data = await response.json();

            updateFooterModels(data.llm_model, data.embed_model);
            closeSettings();
            checkHealth();
        } catch (error) {
            console.error('Failed to reset settings:', error);
        }
    }

    function updateFooterModels(llm, embed) {
        const footer = document.querySelector('.footer');
        const modelSpan = footer.querySelector('span:last-child');
        if (modelSpan) {
            modelSpan.textContent = `${llm.split(':')[0]} + ${embed.split(':')[0]}`;
        }
    }

    settingsBtn.addEventListener('click', openSettings);
    settingsClose.addEventListener('click', closeSettings);
    settingsSave.addEventListener('click', saveSettings);
    settingsReset.addEventListener('click', resetSettings);

    // Close on overlay click
    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) closeSettings();
    });

    // Close on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !settingsModal.classList.contains('hidden')) {
            closeSettings();
        }
    });
}

export { initSettings, loadModels };
