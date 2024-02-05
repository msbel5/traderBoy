// Hardcoded data for testing
var portfolioData = {
    'BTCUSDT': {
        'price': [10, 20, 30, 40, 50],
        'rsi': [30, 40, 50, 60, 70],
        // Add more indicators as needed
    },
    // Add more cryptocurrencies as needed
};

function createChart(containerId, chartData, chartLabel) {
    var ctx = document.getElementById(containerId).getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.map((_, i) => i.toString()),
            datasets: [{
                label: chartLabel,
                data: chartData,
                backgroundColor: 'rgba(0, 123, 255, 0.2)',
                borderColor: 'rgba(0, 123, 255, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function setupCharts() {
    var cryptoSymbols = Object.keys(portfolioData);

    cryptoSymbols.forEach((symbol, index) => {
        var chartContainer = document.createElement('canvas');
        chartContainer.id = symbol + 'Chart';
        document.getElementById('cryptoChart' + (index + 1)).appendChild(chartContainer);
        createChart(chartContainer.id, portfolioData[symbol].price, symbol + ' Price');
    });

    var indicatorsContainer = document.getElementById('indicatorCharts');
    cryptoSymbols.forEach(symbol => {
        Object.keys(portfolioData[symbol]).forEach(indicator => {
            if (indicator !== 'price') {
                var indicatorChartContainer = document.createElement('canvas');
                indicatorChartContainer.id = symbol + indicator + 'Chart';
                indicatorsContainer.appendChild(indicatorChartContainer);
                createChart(indicatorChartContainer.id, portfolioData[symbol][indicator], symbol + ' ' + indicator);
            }
        });
    });
}

window.onload = setupCharts;
