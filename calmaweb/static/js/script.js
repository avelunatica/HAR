// Toggle the navbar menu visibility on mobile
function toggleMenu() {
    var menu = document.querySelector('.navbar-menu-dropdown');
    menu.classList.toggle('active');
}

// Modal de imagen
window.onload = function () {
    var modal = document.getElementById("imageModal");
    var modalImg = document.getElementById("modalImage");
    var captionText = document.getElementById("caption");

    // Selecciona todas las imágenes EXCEPTO las que tengan la clase 'logo-img'
    var allImages = document.querySelectorAll("img:not(.logo-img)");

    allImages.forEach(function (image) {
        image.onclick = function () {
            modal.style.display = "block";
            modalImg.src = this.src;
            captionText.innerHTML = this.alt || "Sin descripción";
        };
    });

    var span = document.getElementsByClassName("close")[0];
    if (span) {
        span.onclick = function () {
            modal.style.display = "none";
        };
    }

    // Activar enlace del navbar
    const navItems = document.querySelectorAll(".navbar-item");
    const currentUrl = window.location.pathname;

    navItems.forEach((item) => {
        if (item.getAttribute("href") === currentUrl) {
            item.classList.add("active");
        } else {
            item.classList.remove("active");
        }
    });
};

// Validación de límite de imágenes
function validateImageLimit() {
    var fileInput = document.getElementById("images");
    var errorText = document.getElementById("imageLimitError");

    if (fileInput && errorText) {
        var fileCount = fileInput.files.length;

        if (fileCount > 5) {
            errorText.style.display = "block";
            fileInput.setCustomValidity("You can only upload up to 5 images.");
        } else {
            errorText.style.display = "none";
            fileInput.setCustomValidity("");
        }
    }
}

document.addEventListener("click", function (event) {
    const toggle = document.querySelector(".navbar-toggle");
    const menu = document.querySelector(".navbar-menu-dropdown");
    const isClickInside = toggle.contains(event.target) || menu.contains(event.target);
  
    if (!isClickInside) {
      menu.classList.remove("active");
    }
  });
  
