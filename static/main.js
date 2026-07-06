document.addEventListener("DOMContentLoaded", () => {
    
    // UI Elements
    const speedVal = document.getElementById("speed-val");
    const ttcVal = document.getElementById("ttc-val");
    const vCount = document.getElementById("v-count");
    const pCount = document.getElementById("p-count");
    const cCount = document.getElementById("c-count");
    const tCount = document.getElementById("t-count");
    const bCount = document.getElementById("b-count");
    const lCount = document.getElementById("l-count");
    
    const gaugeBar = document.getElementById("gauge-bar");
    const gaugeText = document.getElementById("gauge-text");
    const riskPanelBg = document.getElementById("risk-panel-bg");
    const criticalOverlay = document.getElementById("critical-overlay");
    
    const laneStatus = document.getElementById("lane-status");
    const laneConf = document.getElementById("lane-conf");
    const alertsList = document.getElementById("alerts-list");
    
    // Fetch telemetry data periodically
    async function fetchTelemetry() {
        try {
            const response = await fetch('/telemetry');
            if (!response.ok) return;
            const data = await response.json();
            updateUI(data);
        } catch (err) {
            console.error("Telemetry fetch error:", err);
        }
    }
    
    function updateUI(data) {
        // Update basic metrics
        speedVal.innerText = data.speed;
        ttcVal.innerText = data.ttc.toFixed(1);
        
        const m = data.metrics;
        if(m) {
            vCount.innerText = m.vehicles || 0;
            pCount.innerText = m.pedestrians || 0;
            cCount.innerText = m.cyclists || 0;
            tCount.innerText = m.trucks || 0;
            bCount.innerText = m.buses || 0;
            lCount.innerText = m.traffic_lights || 0;
            
            // Risk Gauge
            const risk = m.risk_score || 0;
            gaugeBar.style.width = risk + "%";
            gaugeText.innerText = risk + "%";
            
            // Risk colors and critical overlay
            if (m.risk_status === 'CRITICAL') {
                gaugeText.style.color = 'var(--color-red)';
                riskPanelBg.style.boxShadow = '0 0 30px rgba(239, 68, 68, 0.4)';
                criticalOverlay.classList.remove('hidden');
            } else if (m.risk_status === 'CAUTION') {
                gaugeText.style.color = 'var(--color-yellow)';
                riskPanelBg.style.boxShadow = 'none';
                criticalOverlay.classList.add('hidden');
            } else {
                gaugeText.style.color = 'var(--color-green)';
                riskPanelBg.style.boxShadow = 'none';
                criticalOverlay.classList.add('hidden');
            }
            
            // Lane Assist
            laneStatus.innerText = (m.lane_status || "Stable").toUpperCase();
            laneConf.innerText = m.lane_confidence || "High";
        }
        
        // Update Alerts
        if (data.alerts && data.alerts.length > 0) {
            alertsList.innerHTML = '';
            data.alerts.forEach(alert => {
                const div = document.createElement('div');
                div.className = `alert-item ${alert.level}`;
                div.innerHTML = `
                    <span class="time">● ${alert.time}</span>
                    <span class="msg">${alert.message}</span>
                    <span class="level">${alert.level}</span>
                `;
                alertsList.appendChild(div);
            });
        }
    }
    
    // Poll every 200ms (5 FPS for UI updates is very smooth while video runs natively at 30+ FPS)
    setInterval(fetchTelemetry, 200);
});
