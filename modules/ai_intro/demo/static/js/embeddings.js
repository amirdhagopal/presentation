/**
 * Embeddings Demo Panel
 */

import { API_BASE } from './config.js';

function initEmbeddings() {
    // Compare
    const text1Input = document.getElementById('embed-text1');
    const text2Input = document.getElementById('embed-text2');
    const compareBtn = document.getElementById('embed-compare');
    const compareOutput = document.getElementById('embed-compare-output');

    compareBtn.addEventListener('click', async () => {
        const text1 = text1Input.value.trim();
        const text2 = text2Input.value.trim();
        if (!text1 || !text2) return;

        compareBtn.disabled = true;
        compareOutput.innerHTML = '<span class="thought">Computing embeddings...</span>';

        try {
            const url = `${API_BASE}/api/embeddings/compare?text1=${encodeURIComponent(text1)}&text2=${encodeURIComponent(text2)}`;
            const response = await fetch(url);
            const data = await response.json();

            const percentage = Math.round(data.similarity * 100);
            compareOutput.innerHTML = `
                <div class="similarity-result">
                    <div class="similarity-score">${percentage}%</div>
                    <span class="similarity-label">${data.interpretation}</span>
                    <div class="similarity-bar">
                        <div class="similarity-fill" style="width: ${percentage}%"></div>
                    </div>
                </div>
            `;
        } catch (error) {
            compareOutput.innerHTML = `<span class="error">Error: ${error.message}</span>`;
        } finally {
            compareBtn.disabled = false;
        }
    });

    // Search
    const queryInput = document.getElementById('embed-query');
    const searchBtn = document.getElementById('embed-search');
    const searchOutput = document.getElementById('embed-search-output');

    searchBtn.addEventListener('click', async () => {
        const query = queryInput.value.trim();
        if (!query) return;

        searchBtn.disabled = true;
        searchOutput.innerHTML = '<span class="thought">Searching...</span>';

        try {
            const url = `${API_BASE}/api/embeddings/search?query=${encodeURIComponent(query)}`;
            const response = await fetch(url);
            const data = await response.json();

            let html = '<ul class="search-results">';
            for (const result of data.results) {
                const percentage = Math.round(result.similarity * 100);
                html += `
                    <li class="search-result">
                        <span class="score">${percentage}%</span>
                        <p class="text">${result.document}</p>
                    </li>
                `;
            }
            html += '</ul>';
            searchOutput.innerHTML = html;
        } catch (error) {
            searchOutput.innerHTML = `<span class="error">Error: ${error.message}</span>`;
        } finally {
            searchBtn.disabled = false;
        }
    });
}

export { initEmbeddings };
