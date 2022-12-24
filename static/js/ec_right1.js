var ec_right1 = echarts.init(document.getElementById('r1'),"dark");

var option_right1 = {

	backgroundColor: '#4F4F4F',
	title: {
		text: '非湖北地区城市确诊TOP3',
		textStyle: {
			color: 'white'
		},
		left: 'left'
	},
	grid: {
	// 	left: 50,
	// 	top: 50,
		right: 0,
		width: '70%',
	// 	height: 320,
	},
	color: ['#3398DB'],
	tooltip: {
		trigger: 'axis',
		axisPointer: {
			type: 'shadow'
		}
	},
	//全局字体样式
	// textStyle: {
	// 	fontFamily: 'PingFangSC-Medium',
	// 	fontSize: 12,
	// 	color: '#858E96',
	// 	lineHeight: 12
	// },
	xAxis: {
		type: 'category',
		data: []
	},
	yAxis: {
		type: 'value',
		//坐标轴刻度设置
		},
	series: [{
		data: [],
		type: 'bar',
		barMaxWidth: "50%"
	}]
};
ec_right1.setOption(option_right1)
