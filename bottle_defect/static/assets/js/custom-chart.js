var bar1 = document.getElementById('bar1');
new Chart(bar1, {
  type: 'bar',
  data: {
    labels: ['Defective', 'Non-Defective'],
    datasets: [{
      data: ['prediction'],
      backgroundColor: [
        'green(1)',
        'blue(0)',
    ],
    borderColor: [
         'green(1)',
         'blue(0)',
    ],
      borderWidth: 1
    }]
  },
  options: {
    plugins: {
      legend: {
          display: false
      },
    }
  }
});


//var bar2 = document.getElementById('bar2');
//new Chart(bar2, {
//  type: 'bar',
//  data: {
//    labels: ['Blue', 'Red', 'Yellow', 'Green'],
//    datasets: [{
//      data: [40, 60, 30, 20],
//      backgroundColor: [
//        'rgba(54, 162, 235, 0.2)',
//        'rgba(255, 99, 132, 0.2)',
//        'rgba(255, 206, 86, 0.2)',
//        'rgba(75, 192, 192, 0.2)',
//    ],
//    borderColor: [
//        'rgba(54, 162, 235, 1)',
//        'rgba(255,99,132,1)',
//        'rgba(255, 206, 86, 1)',
//        'rgba(75, 192, 192, 1)',
//    ],
//      borderWidth: 1
//    }]
//  },
//  options: {
//    plugins: {
//      legend: {
//          display: false
//      },
//    }
//  }
//});
//
//var bar3 = document.getElementById('bar3');
//new Chart(bar3, {
//  type: 'bar',
//  data: {
//    labels: ['Blue', 'Red', 'Yellow', 'Green'],
//    datasets: [{
//      data: [40, 60, 30, 20],
//      backgroundColor: [
//        'rgba(54, 162, 235, 0.2)',
//        'rgba(255, 99, 132, 0.2)',
//        'rgba(255, 206, 86, 0.2)',
//        'rgba(75, 192, 192, 0.2)',
//    ],
//    borderColor: [
//        'rgba(54, 162, 235, 1)',
//        'rgba(255,99,132,1)',
//        'rgba(255, 206, 86, 1)',
//        'rgba(75, 192, 192, 1)',
//    ],
//      borderWidth: 1
//    }]
//  },
//  options: {
//    plugins: {
//      legend: {
//          display: false
//      },
//    }
//  }
//});