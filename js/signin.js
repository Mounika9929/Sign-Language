document.getElementById("signin-form").addEventListener("submit", async function(event) {
    event.preventDefault();

    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;
    const errorMessage = document.getElementById("error-message");

    try {
        const response = await fetch("/api/signin", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password }),
        });

        const data = await response.json();
        if (response.ok) {
            alert("Sign in successful!");
            window.location.href = "/dashboard";  // Redirect on success
        } else {
            errorMessage.textContent = data.error;
        }
    } catch (error) {
        errorMessage.textContent = "Server error. Please try again.";
    }
});
