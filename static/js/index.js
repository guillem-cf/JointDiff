window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');
    
    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');
    
    if (carouselVideos.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });
    
    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

// ── Scrollytelling ──────────────────────────────────────
// Configure your slides here. Order determines scroll order.
// type: 'pdf' for static PDF slides, 'img' for GIF/PNG/JPG images.
var SLIDES = [
  { src: 'static/slides/01-intro.gif', type: 'img' },
  { src: 'static/slides/02-motivation.gif', type: 'img' },
  { src: 'static/slides/03-framework.pdf', type: 'pdf' },
  { src: 'static/slides/04-denoiser.pdf', type: 'pdf' },
  { src: 'static/slides/05-crossguid.pdf', type: 'pdf' },
  { src: 'static/slides/06-futuregen.pdf', type: 'pdf' },
  { src: 'static/slides/07-controlgen.pdf', type: 'pdf' },
  // { src: 'static/slides/08-text-guidance.pdf', type: 'pdf' },
  // { src: 'static/slides/09-results.gif', type: 'img' },
  // { src: 'static/slides/10-qualitative.gif', type: 'img' },
  // { src: 'static/slides/11-ablation.pdf', type: 'pdf' },
  // { src: 'static/slides/12-conclusion.pdf', type: 'pdf' }
];

// PDF.js worker setup
if (typeof pdfjsLib !== 'undefined') {
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
}

function renderPdfToCanvas(canvas, url) {
  if (canvas.dataset.rendered) return Promise.resolve();
  canvas.dataset.rendered = 'true';
  return pdfjsLib.getDocument(url).promise.then(function(pdf) {
    return pdf.getPage(1);
  }).then(function(page) {
    var scale = 2;
    var viewport = page.getViewport({ scale: scale });
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    return page.render({
      canvasContext: canvas.getContext('2d'),
      viewport: viewport
    }).promise;
  }).catch(function(err) {
    console.warn('PDF render failed for ' + url + ':', err);
  });
}

function initScrollytelling() {
  var container = document.getElementById('scrollytelling');
  if (!container || SLIDES.length === 0) return;

  var stack = document.getElementById('scrollytelling-stack');
  var dotsContainer = document.getElementById('scrollytelling-dots');
  var spacersContainer = document.getElementById('scrollytelling-spacers');
  var progressTotal = container.querySelector('.scrollytelling__progress-total');
  var progressCurrent = container.querySelector('.scrollytelling__progress-current');

  // Set total
  if (progressTotal) progressTotal.textContent = SLIDES.length;

  // Build slide elements, dots, and spacers
  var slideElements = [];
  SLIDES.forEach(function(slide, i) {
    var num = i + 1;
    var el;

    if (slide.type === 'pdf') {
      el = document.createElement('canvas');
      el.className = 'scrollytelling__slide' + (i === 0 ? ' scrollytelling__slide--active' : '');
      el.dataset.slide = num;
      el.dataset.pdf = slide.src;
    } else {
      el = document.createElement('img');
      el.className = 'scrollytelling__slide' + (i === 0 ? ' scrollytelling__slide--active' : '');
      el.dataset.slide = num;
      el.alt = 'Slide ' + num;
      if (i === 0) {
        el.src = slide.src;
      } else {
        el.dataset.src = slide.src;
        el.loading = 'lazy';
      }
    }

    stack.appendChild(el);
    slideElements.push(el);

    // Dot
    var dot = document.createElement('button');
    dot.className = 'scrollytelling__dot' + (i === 0 ? ' scrollytelling__dot--active' : '');
    dot.dataset.slide = num;
    dot.setAttribute('aria-label', 'Go to slide ' + num);
    dotsContainer.appendChild(dot);

    // Spacer
    var spacer = document.createElement('div');
    spacer.className = 'scrollytelling__spacer';
    spacer.dataset.step = num;
    spacersContainer.appendChild(spacer);
  });

  // Render first PDF slide immediately
  if (SLIDES[0].type === 'pdf') {
    renderPdfToCanvas(slideElements[0], SLIDES[0].src);
  }

  var currentStep = 1;
  var allDots = dotsContainer.querySelectorAll('.scrollytelling__dot');
  var allSpacers = spacersContainer.querySelectorAll('.scrollytelling__spacer');

  function activateStep(stepNum) {
    if (stepNum === currentStep) return;
    currentStep = stepNum;

    // Crossfade slides
    slideElements.forEach(function(el, i) {
      var isTarget = (i + 1) === stepNum;
      el.classList.toggle('scrollytelling__slide--active', isTarget);
    });

    // Update dots
    allDots.forEach(function(dot) {
      dot.classList.toggle('scrollytelling__dot--active',
        parseInt(dot.dataset.slide, 10) === stepNum);
    });

    // Update progress
    if (progressCurrent) progressCurrent.textContent = stepNum;

    // Lazy load current and adjacent slides
    [stepNum - 2, stepNum - 1, stepNum, stepNum + 1].forEach(function(n) {
      var idx = n - 1;
      if (idx < 0 || idx >= SLIDES.length) return;
      var el = slideElements[idx];
      var slide = SLIDES[idx];

      if (slide.type === 'pdf') {
        renderPdfToCanvas(el, slide.src);
      } else if (el.dataset.src) {
        el.src = el.dataset.src;
        delete el.dataset.src;
      }
    });
  }

  // IntersectionObserver for spacers
  var stepObserver = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        var stepNum = parseInt(entry.target.dataset.step, 10);
        activateStep(stepNum);
      }
    });
  }, {
    rootMargin: '-40% 0px -40% 0px',
    threshold: 0
  });

  allSpacers.forEach(function(spacer) {
    stepObserver.observe(spacer);
  });

  // Dot click navigation
  allDots.forEach(function(dot) {
    dot.addEventListener('click', function() {
      var stepNum = parseInt(dot.dataset.slide, 10);
      var targetSpacer = spacersContainer.querySelector(
        '[data-step="' + stepNum + '"]'
      );
      if (targetSpacer) {
        targetSpacer.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    });
  });
}

document.addEventListener('DOMContentLoaded', function() {
  initScrollytelling();
});

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
    }

	// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();
    
    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();

})
