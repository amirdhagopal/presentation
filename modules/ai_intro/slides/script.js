/**
 * AI Concepts Presentation - Interactive Controls
 */

(function () {
    'use strict';

    // ========================================
    // State Management
    // ========================================
    const state = {
        currentSlide: 1,
        totalSlides: 8,
        isAnimating: false,
        touchStartX: 0,
        touchEndX: 0
    };

    // ========================================
    // DOM Elements
    // ========================================
    const elements = {
        slides: null,
        progressBar: null,
        currentSlideEl: null,
        totalSlidesEl: null,
        prevBtn: null,
        nextBtn: null
    };

    // ========================================
    // Initialization
    // ========================================
    function init() {
        // Cache DOM elements
        elements.slides = document.querySelectorAll('.slide');
        elements.progressBar = document.getElementById('progressBar');
        elements.currentSlideEl = document.getElementById('currentSlide');
        elements.totalSlidesEl = document.getElementById('totalSlides');
        elements.prevBtn = document.getElementById('prevBtn');
        elements.nextBtn = document.getElementById('nextBtn');

        // Update total slides count
        state.totalSlides = elements.slides.length;
        elements.totalSlidesEl.textContent = state.totalSlides;

        // Make slides focusable for keyboard scrolling
        elements.slides.forEach(slide => {
            slide.setAttribute('tabindex', '0');
        });

        // Bind event listeners
        bindEvents();

        // Initial state
        updateUI();

        // Check for hash in URL
        const hash = window.location.hash;
        if (hash && hash.startsWith('#slide-')) {
            const slideNum = parseInt(hash.replace('#slide-', ''));
            if (slideNum >= 1 && slideNum <= state.totalSlides) {
                goToSlide(slideNum, false);
            }
        }

        // Focus the active slide for immediate keyboard scrolling
        focusActiveSlide();

        console.log('🚀 AI Concepts Presentation initialized');
    }

    // ========================================
    // Event Binding
    // ========================================
    function bindEvents() {
        // Keyboard navigation
        document.addEventListener('keydown', handleKeyboard);

        // Navigation buttons
        elements.prevBtn.addEventListener('click', () => navigate(-1));
        elements.nextBtn.addEventListener('click', () => navigate(1));

        // Touch events for mobile swipe
        document.addEventListener('touchstart', handleTouchStart, { passive: true });
        document.addEventListener('touchend', handleTouchEnd, { passive: true });

        // Window resize
        window.addEventListener('resize', debounce(handleResize, 250));

        // Prevent context menu on long press (mobile)
        document.addEventListener('contextmenu', (e) => {
            if (e.target.closest('.nav-btn')) {
                e.preventDefault();
            }
        });
    }

    // ========================================
    // Navigation
    // ========================================
    function navigate(direction) {
        if (state.isAnimating) return;

        const newSlide = state.currentSlide + direction;

        if (newSlide >= 1 && newSlide <= state.totalSlides) {
            goToSlide(newSlide);
        }
    }

    function goToSlide(slideNumber, animate = true) {
        if (slideNumber === state.currentSlide) return;
        if (slideNumber < 1 || slideNumber > state.totalSlides) return;

        state.isAnimating = true;

        const currentSlideEl = elements.slides[state.currentSlide - 1];
        const nextSlideEl = elements.slides[slideNumber - 1];

        // Add exit class to current slide
        currentSlideEl.classList.add('exit');
        currentSlideEl.classList.remove('active');

        // Activate new slide
        nextSlideEl.classList.add('active');

        // Reset scroll position for new slide
        nextSlideEl.scrollTop = 0;

        // Update state
        state.currentSlide = slideNumber;

        // Update UI
        updateUI();

        // Update URL hash
        history.replaceState(null, null, `#slide-${slideNumber}`);

        // Reset animation lock after transition and focus the slide
        setTimeout(() => {
            currentSlideEl.classList.remove('exit');
            state.isAnimating = false;
            focusActiveSlide();
        }, animate ? 500 : 0);
    }

    // Focus the active slide so arrow keys work for scrolling
    function focusActiveSlide() {
        const activeSlide = elements.slides[state.currentSlide - 1];
        if (activeSlide) {
            activeSlide.focus({ preventScroll: true });
        }
    }

    // ========================================
    // UI Updates
    // ========================================
    function updateUI() {
        // Update slide counter
        elements.currentSlideEl.textContent = state.currentSlide;

        // Update progress bar
        const progress = (state.currentSlide / state.totalSlides) * 100;
        elements.progressBar.style.width = `${progress}%`;

        // Update navigation buttons
        elements.prevBtn.disabled = state.currentSlide === 1;
        elements.nextBtn.disabled = state.currentSlide === state.totalSlides;

        // Trigger animations for current slide
        triggerSlideAnimations();
    }

    function triggerSlideAnimations() {
        const currentSlideEl = elements.slides[state.currentSlide - 1];
        const animatedElements = currentSlideEl.querySelectorAll('.animate-in');

        // Reset animations
        animatedElements.forEach(el => {
            el.style.animation = 'none';
            el.offsetHeight; // Trigger reflow
            el.style.animation = null;
        });
    }

    // ========================================
    // Keyboard Handler
    // ========================================
    function handleKeyboard(e) {
        // Ignore if user is typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }

        switch (e.key) {
            case 'ArrowRight':
            case ' ':
            case 'PageDown':
                e.preventDefault();
                navigate(1);
                break;

            case 'ArrowLeft':
            case 'PageUp':
                e.preventDefault();
                navigate(-1);
                break;

            // ArrowUp and ArrowDown are NOT handled here - they scroll the content naturally

            case 'Home':
            case '[':
                e.preventDefault();
                goToSlide(1);
                break;

            case 'End':
            case ']':
                e.preventDefault();
                goToSlide(state.totalSlides);
                break;

            case 'f':
            case 'F':
                e.preventDefault();
                toggleFullscreen();
                break;

            // Number keys for direct slide access
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                const num = parseInt(e.key);
                if (num <= state.totalSlides) {
                    e.preventDefault();
                    goToSlide(num);
                }
                break;
        }
    }

    // ========================================
    // Touch Handlers (Swipe Navigation)
    // ========================================
    function handleTouchStart(e) {
        state.touchStartX = e.changedTouches[0].screenX;
    }

    function handleTouchEnd(e) {
        state.touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    }

    function handleSwipe() {
        const swipeThreshold = 50;
        const diff = state.touchStartX - state.touchEndX;

        if (Math.abs(diff) > swipeThreshold) {
            if (diff > 0) {
                // Swipe left - next slide
                navigate(1);
            } else {
                // Swipe right - previous slide
                navigate(-1);
            }
        }
    }

    // ========================================
    // Fullscreen Toggle
    // ========================================
    function toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(err => {
                console.log('Fullscreen error:', err);
            });
        } else {
            document.exitFullscreen();
        }
    }

    // ========================================
    // Resize Handler
    // ========================================
    function handleResize() {
        // Recalculate any size-dependent elements
        // Currently not needed, but placeholder for future enhancements
    }

    // ========================================
    // Utility Functions
    // ========================================
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // ========================================
    // Initialize on DOM Ready
    // ========================================
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // ========================================
    // Expose API for debugging/extensions
    // ========================================
    window.presentation = {
        goToSlide,
        navigate,
        getState: () => ({ ...state }),
        toggleFullscreen
    };

})();
