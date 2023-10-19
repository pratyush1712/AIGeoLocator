var map = L.map("map").setView([42.4072, -71.3824], 8);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);

var drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

var drawControl = new L.Control.Draw({
    draw: {
        rectangle: false,
        polyline: false,
        polygon: false,
        circle: false,
        marker: false,
        circlemarker: false,
    },
    edit: {
        featureGroup: drawnItems,
        edit: false,
        remove: false,
    },
});
map.addControl(drawControl);

var heatmapLayer;

let thresholdSlider = document.getElementById("threshold");
let thresholdValue = document.getElementById("thresholdValue");
let kSlider = document.getElementById("k");
let kValue = document.getElementById("kValue");

let form = document.getElementById("search");
let slidingTimeout;

thresholdSlider.disabled = true;
kSlider.disabled = true;

form.addEventListener("submit", sendRequest);

thresholdSlider.addEventListener("input", function () {
    thresholdValue.textContent = this.value;
});

thresholdSlider.addEventListener("change", function () {
    if (slidingTimeout) {
        clearTimeout(slidingTimeout);
    }

    slidingTimeout = setTimeout(function () {
        sendRequest();
    }, 1000);
});

kSlider.addEventListener("input", function () {
    kValue.textContent = this.value;
});

kSlider.addEventListener("change", function () {
    if (slidingTimeout) {
        clearTimeout(slidingTimeout);
    }

    slidingTimeout = setTimeout(function () {
        sendRequest();
    }, 1000);
});

function sendRequest(e) {
    if (e) e.preventDefault();
    document.getElementById("loading-icon").style.display = "flex";
    var query = document.getElementById("userInput").value;
    var threshold = document.getElementById("threshold").value;
    var k = document.getElementById("k").value;
    if (heatmapLayer) {
        map.removeLayer(heatmapLayer);
    }

    var coordinates = {
        southWest: {
            lat: 0,
            lng: 0,
        },
        northEast: {
            lat: 0,
            lng: 0,
        },
    };

    var thresh_arg = threshold !== "" ? `&thresh=${threshold}` : "";
    var k_arg = k !== "" ? `&k=${k}` : "";
    fetch(`/classified-points?query=${query}${thresh_arg}${k_arg}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(coordinates),
        })
        .then((response) => response.json())
        .then((data) => {
            map.eachLayer(function (layer) {
                if (layer instanceof L.Marker) {
                    map.removeLayer(layer);
                }
            });

            var heatmapData = data.blue_coords.map(function (point) {
                return [point[0], point[1]];
            });

            heatmapLayer = new L.heatLayer(heatmapData, (value = data.confidences), {
                radius: 2,
                blur: 0,
                maxZoom: 18,
                max: 1.0,
                gradient: {
                    0.92: "blue",
                    0.94: "green",
                    0.96: "yellow",
                    0.98: "orange",
                    1.0: "red",
                },
            }).addTo(map);

            data.top_locs.forEach((loc) => {
                var redIcon = L.divIcon({
                    className: "custom-icon",
                    html: `<div class="marker-label">${loc[2]}</div><img src="https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png" style="display:block;"/>`,
                    iconSize: [25, 41],
                    iconAnchor: [12, 41],
                    popupAnchor: [0, -20],
                });
                var marker = L.marker([loc[1][1], loc[1][0]], {
                    icon: redIcon,
                }).addTo(map);

                var imageUrl = `https://research.cs.cornell.edu/caco/data/graft/MA/${loc[0]}`;
                var popupContent = `<img src="${imageUrl}" class="small-image" />`;
                marker.bindPopup(popupContent, {
                    minWidth: "fit-content",
                    minHeight: "fit-content",
                });

                // Show the popup (with the image) when hovering over the marker
                marker.on("mouseover", function (e) {
                    this.openPopup();
                });
                marker.on("mouseout", function (e) {
                    this.closePopup();
                });
            });

            document.getElementById("loading-icon").style.display = "none";
            thresholdSlider.disabled = false;
            kSlider.disabled = false;
        })
        .catch((error) => {
            console.error("Error:", error);
            document.getElementById("loading-icon").style.display = "none";
            thresholdSlider.disabled = false;
            kSlider.disabled = false;
        });
}

function toggleHeatmap() {
    if (heatmapLayer) {
        if (!map.hasLayer(heatmapLayer)) {
            heatmapLayer.addTo(map);
            thresholdSlider.disabled = false;
            kSlider.disabled = false;
        } else {
            map.removeLayer(heatmapLayer);
            thresholdSlider.disabled = true;
            kSlider.disabled = true;
        }
    }
}

function resetPage() {
    location.reload();
}

L.Control.geocoder().addTo(map);