const form = document.getElementById("upload-form");
const outputDiv = document.getElementById("output");
const predictionsTable = document.getElementById("predictions-table");
const predictionsHeader = document.getElementById("predictions-header");
const predictionsBody = document.getElementById("predictions-body");
const downloadButton = document.getElementById("download-btn");

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("file");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select a file!");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://localhost:8080/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }

        const result = await response.json();

        // Clear previous table content
        predictionsHeader.innerHTML = "";
        predictionsBody.innerHTML = "";
        downloadButton.classList.add("hidden"); // Hide download button initially

        if (result.error) {
            predictionsTable.style.display = "none";
            predictionsBody.innerHTML = `<tr><td colspan="7">Error: ${result.error}</td></tr>`;
        } else {
            // Display table headers dynamically
            const headers = Object.keys(result.predictions[0]);
            const headerRow = document.createElement("tr");
            headers.forEach(header => {
                const th = document.createElement("th");
                th.textContent = header;
                headerRow.appendChild(th);
            });
            predictionsHeader.appendChild(headerRow);

            // Populate table body with prediction rows
            result.predictions.forEach(prediction => {
                const row = document.createElement("tr");
                Object.entries(prediction).forEach(([key, value]) => {
                    const cell = document.createElement("td");
                    cell.textContent = value;

                    // Style based on "Attack Type" ONLY
                    if (key === "Attack Type") {
                        cell.style.backgroundColor = value === "BENIGN" ? "green" : "red";
                        cell.style.color = "white";
                    }

                    row.appendChild(cell);
                });
                predictionsBody.appendChild(row);
            });

            // Enable download button
            downloadButton.onclick = () => downloadPredictions(result.predictions);
            downloadButton.classList.remove("hidden");
        }

        predictionsTable.style.display = "table";
        outputDiv.classList.remove("hidden");
    } catch (error) {
        predictionsBody.innerHTML = `<tr><td colspan="7">Error: ${error.message}</td></tr>`;
        outputDiv.classList.remove("hidden");
    }
});


function downloadPredictions(predictions) {
    const headers = Object.keys(predictions[0]);
    const rows = predictions.map(prediction => headers.map(header => prediction[header]).join(","));
    const csvContent = [headers.join(","), ...rows].join("\n");
    const encodedUri = encodeURI("data:text/csv;charset=utf-8," + csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "predictions.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
