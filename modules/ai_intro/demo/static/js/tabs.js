/**
 * Tab Navigation
 */

function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    const panels = document.querySelectorAll('.panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.dataset.tab;

            // Update tabs
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Update panels
            panels.forEach(p => p.classList.remove('active'));
            document.getElementById(`panel-${tabId}`).classList.add('active');
        });
    });
}

function initSubTabs() {
    const subTabs = document.querySelectorAll('.sub-tab');

    subTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const subtabId = tab.dataset.subtab;
            const parent = tab.closest('.panel');

            // Update sub-tabs
            parent.querySelectorAll('.sub-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Update sub-panels
            parent.querySelectorAll('.sub-panel').forEach(p => p.classList.remove('active'));
            document.getElementById(`subpanel-${subtabId}`).classList.add('active');
        });
    });
}

export { initTabs, initSubTabs };
