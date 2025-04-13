document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("form");
  
    if (form) {
      form.addEventListener("submit", () => {
        const btn = form.querySelector("button");
        btn.disabled = true;
        btn.innerHTML = "⏳ Predicting...";
      });
    }
  
    // Fade in prediction result
    const alertBox = document.querySelector(".alert");
    if (alertBox) {
      alertBox.style.opacity = 0;
      alertBox.style.transition = "opacity 1s ease-in-out";
      setTimeout(() => {
        alertBox.style.opacity = 1;
      }, 300);
    }
  
    // Scroll reveal effect (simple)
    const elements = document.querySelectorAll(".form-control, .form-select, button");
    elements.forEach((el, index) => {
      el.style.opacity = 0;
      el.style.transform = "translateY(20px)";
      el.style.transition = `opacity 0.6s ease ${index * 0.05}s, transform 0.6s ease ${index * 0.05}s`;
      setTimeout(() => {
        el.style.opacity = 1;
        el.style.transform = "translateY(0)";
      }, 200);
    });
  });




  document.addEventListener("DOMContentLoaded", () => {
    const neurons = document.querySelectorAll(".neuron");
  
    neurons.forEach(neuron => {
      neuron.addEventListener("mouseenter", () => {
        neuron.setAttribute("fill", "#1abc9c");
      });
      neuron.addEventListener("mouseleave", () => {
        neuron.setAttribute("fill", neuron.dataset.original || neuron.getAttribute("fill"));
      });
    });
  });
  
  