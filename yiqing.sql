-- ----------------------------
-- Table structure for details
-- ----------------------------
DROP TABLE IF EXISTS `details`;
CREATE TABLE `details`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `update_time` datetime NULL DEFAULT NULL COMMENT '数据最后更新时间',
  `province` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '省',
  `city` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '市',
  `confirm` int(11) NULL DEFAULT NULL COMMENT '累计确诊',
  `confirm_add` int(11) NULL DEFAULT NULL COMMENT '新增治愈',
  `heal` int(11) NULL DEFAULT NULL COMMENT '累计治愈',
  `dead` int(11) NULL DEFAULT NULL COMMENT '累计死亡',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 7920 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of details
-- ----------------------------
INSERT INTO `details` VALUES (7441, '2022-02-10 14:30:59', '台湾', '地区待确认', 19376, 137, 13742, 851);
INSERT INTO `details` VALUES (7442, '2022-02-10 14:30:59', '香港', '地区待确认', 15811, 635, 13436, 215);
INSERT INTO `details` VALUES (7443, '2022-02-10 14:30:59', '浙江', '杭州', 328, 0, 187, 0);
INSERT INTO `details` VALUES (7444, '2022-02-10 14:30:59', '浙江', '宁波', 269, 0, 159, 0);
INSERT INTO `details` VALUES (7445, '2022-02-10 14:30:59', '浙江', '境外输入', 375, 1, 268, 0);
INSERT INTO `details` VALUES (7446, '2022-02-10 14:30:59', '浙江', '绍兴', 429, 0, 419, 0);
INSERT INTO `details` VALUES (7447, '2022-02-10 14:30:59', '浙江', '金华', 57, 0, 55, 0);
INSERT INTO `details` VALUES (7448, '2022-02-10 14:30:59', '浙江', '衢州', 14, 0, 14, 0);
INSERT INTO `details` VALUES (7449, '2022-02-10 14:30:59', '浙江', '台州', 147, 0, 147, 0);
INSERT INTO `details` VALUES (7450, '2022-02-10 14:30:59', '浙江', '地区待确认', 0, 0, 0, 0);
INSERT INTO `details` VALUES (7451, '2022-02-10 14:30:59', '浙江', '省十里丰监狱', 36, 0, 36, 0);
INSERT INTO `details` VALUES (7452, '2022-02-10 14:30:59', '浙江', '丽水', 17, 0, 17, 0);
INSERT INTO `details` VALUES (7453, '2022-02-10 14:30:59', '浙江', '舟山', 10, 0, 10, 0);
INSERT INTO `details` VALUES (7454, '2022-02-10 14:30:59', '浙江', '嘉兴', 46, 0, 46, 0);
INSERT INTO `details` VALUES (7455, '2022-02-10 14:30:59', '浙江', '湖州', 10, 0, 10, 0);
INSERT INTO `details` VALUES (7456, '2022-02-10 14:30:59', '浙江', '温州', 504, 0, 503, 1);
INSERT INTO `details` VALUES (7457, '2022-02-10 14:30:59', '广西', '百色', 189, 7, 3, 0);
INSERT INTO `details` VALUES (7458, '2022-02-10 14:30:59', '广西', '境外输入', 451, 1, 356, 0);
INSERT INTO `details` VALUES (7459, '2022-02-10 14:30:59', '广西', '崇左', 1, 0, 0, 0);
INSERT INTO `details` VALUES (7460, '2022-02-10 14:30:59', '广西', '防城港', 39, 0, 38, 0);
INSERT INTO `details` VALUES (7461, '2022-02-10 14:30:59', '广西', '柳州', 24, 0, 24, 0);
INSERT INTO `details` VALUES (7462, '2022-02-10 14:30:59', '广西', '南宁', 56, 0, 56, 0);
INSERT INTO `details` VALUES (7463, '2022-02-10 14:30:59', '广西', '北海', 44, 0, 43, 1);
INSERT INTO `details` VALUES (7464, '2022-02-10 14:30:59', '广西', '来宾', 11, 0, 11, 0);
INSERT INTO `details` VALUES (7465, '2022-02-10 14:30:59', '广西', '贺州', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7466, '2022-02-10 14:30:59', '广西', '河池', 28, 0, 27, 1);
INSERT INTO `details` VALUES (7467, '2022-02-10 14:30:59', '广西', '桂林', 32, 0, 32, 0);
INSERT INTO `details` VALUES (7468, '2022-02-10 14:30:59', '广西', '玉林', 11, 0, 11, 0);
INSERT INTO `details` VALUES (7469, '2022-02-10 14:30:59', '广西', '钦州', 8, 0, 8, 0);
INSERT INTO `details` VALUES (7470, '2022-02-10 14:30:59', '广西', '贵港', 8, 0, 8, 0);
INSERT INTO `details` VALUES (7471, '2022-02-10 14:30:59', '广西', '梧州', 5, 0, 5, 0);
INSERT INTO `details` VALUES (7472, '2022-02-10 14:30:59', '广西', '地区待确认', 0, 0, 1, 0);
INSERT INTO `details` VALUES (7473, '2022-02-10 14:30:59', '河南', '安阳', 522, 0, 346, 0);
INSERT INTO `details` VALUES (7474, '2022-02-10 14:30:59', '河南', '郑州', 488, 0, 337, 5);
INSERT INTO `details` VALUES (7475, '2022-02-10 14:30:59', '河南', '许昌', 405, 0, 258, 1);
INSERT INTO `details` VALUES (7476, '2022-02-10 14:30:59', '河南', '周口', 101, 0, 76, 1);
INSERT INTO `details` VALUES (7477, '2022-02-10 14:30:59', '河南', '境外输入', 154, 0, 138, 0);
INSERT INTO `details` VALUES (7478, '2022-02-10 14:30:59', '河南', '洛阳', 41, 0, 30, 1);
INSERT INTO `details` VALUES (7479, '2022-02-10 14:30:59', '河南', '信阳', 277, 0, 272, 2);
INSERT INTO `details` VALUES (7480, '2022-02-10 14:30:59', '河南', '商丘', 109, 0, 104, 3);
INSERT INTO `details` VALUES (7481, '2022-02-10 14:30:59', '河南', '平顶山', 59, 0, 57, 1);
INSERT INTO `details` VALUES (7482, '2022-02-10 14:30:59', '河南', '三门峡', 8, 0, 6, 1);
INSERT INTO `details` VALUES (7483, '2022-02-10 14:30:59', '河南', '鹤壁', 19, 0, 19, 0);
INSERT INTO `details` VALUES (7484, '2022-02-10 14:30:59', '河南', '驻马店', 143, 0, 143, 0);
INSERT INTO `details` VALUES (7485, '2022-02-10 14:30:59', '河南', '开封', 33, 0, 33, 0);
INSERT INTO `details` VALUES (7486, '2022-02-10 14:30:59', '河南', '漯河', 36, 0, 36, 0);
INSERT INTO `details` VALUES (7487, '2022-02-10 14:30:59', '河南', '南阳', 156, 0, 153, 3);
INSERT INTO `details` VALUES (7488, '2022-02-10 14:30:59', '河南', '济源示范区', 5, 0, 5, 0);
INSERT INTO `details` VALUES (7489, '2022-02-10 14:30:59', '河南', '新乡', 57, 0, 54, 3);
INSERT INTO `details` VALUES (7490, '2022-02-10 14:30:59', '河南', '焦作', 32, 0, 31, 1);
INSERT INTO `details` VALUES (7491, '2022-02-10 14:30:59', '河南', '濮阳', 17, 0, 17, 0);
INSERT INTO `details` VALUES (7492, '2022-02-10 14:30:59', '河南', '地区待确认', 0, 0, 249, 0);
INSERT INTO `details` VALUES (7493, '2022-02-10 14:30:59', '上海', '境外输入', 3466, 8, 3251, 0);
INSERT INTO `details` VALUES (7494, '2022-02-10 14:30:59', '上海', '奉贤', 11, 0, 9, 0);
INSERT INTO `details` VALUES (7495, '2022-02-10 14:30:59', '上海', '境外来沪', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7496, '2022-02-10 14:30:59', '上海', '地区待确认', 0, 0, 0, 0);
INSERT INTO `details` VALUES (7497, '2022-02-10 14:30:59', '上海', '浦东', 82, 0, 81, 1);
INSERT INTO `details` VALUES (7498, '2022-02-10 14:30:59', '上海', '青浦', 7, 0, 7, 0);
INSERT INTO `details` VALUES (7499, '2022-02-10 14:30:59', '上海', '松江', 16, 0, 16, 0);
INSERT INTO `details` VALUES (7500, '2022-02-10 14:30:59', '上海', '宝山', 26, 0, 25, 1);
INSERT INTO `details` VALUES (7501, '2022-02-10 14:30:59', '上海', '黄浦', 22, 0, 22, 0);
INSERT INTO `details` VALUES (7502, '2022-02-10 14:30:59', '上海', '长宁', 14, 0, 14, 0);
INSERT INTO `details` VALUES (7503, '2022-02-10 14:30:59', '上海', '静安', 19, 0, 18, 1);
INSERT INTO `details` VALUES (7504, '2022-02-10 14:30:59', '上海', '外地来沪', 113, 0, 112, 1);
INSERT INTO `details` VALUES (7505, '2022-02-10 14:30:59', '上海', '嘉定', 9, 0, 7, 2);
INSERT INTO `details` VALUES (7506, '2022-02-10 14:30:59', '上海', '徐汇', 18, 0, 17, 1);
INSERT INTO `details` VALUES (7507, '2022-02-10 14:30:59', '上海', '闵行', 19, 0, 19, 0);
INSERT INTO `details` VALUES (7508, '2022-02-10 14:30:59', '上海', '普陀', 11, 0, 11, 0);
INSERT INTO `details` VALUES (7509, '2022-02-10 14:30:59', '上海', '杨浦', 9, 0, 9, 0);
INSERT INTO `details` VALUES (7510, '2022-02-10 14:30:59', '上海', '虹口', 7, 0, 7, 0);
INSERT INTO `details` VALUES (7511, '2022-02-10 14:30:59', '上海', '金山', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7512, '2022-02-10 14:30:59', '上海', '崇明', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7513, '2022-02-10 14:30:59', '广东', '境外输入', 2201, 9, 2030, 0);
INSERT INTO `details` VALUES (7514, '2022-02-10 14:30:59', '广东', '珠海', 134, 0, 98, 1);
INSERT INTO `details` VALUES (7515, '2022-02-10 14:30:59', '广东', '深圳', 464, 0, 428, 3);
INSERT INTO `details` VALUES (7516, '2022-02-10 14:30:59', '广东', '东莞', 128, 0, 101, 1);
INSERT INTO `details` VALUES (7517, '2022-02-10 14:30:59', '广东', '云浮', 7, 0, 0, 0);
INSERT INTO `details` VALUES (7518, '2022-02-10 14:30:59', '广东', '惠州', 68, 0, 62, 0);
INSERT INTO `details` VALUES (7519, '2022-02-10 14:30:59', '广东', '广州', 527, 0, 522, 1);
INSERT INTO `details` VALUES (7520, '2022-02-10 14:30:59', '广东', '中山', 71, 0, 68, 0);
INSERT INTO `details` VALUES (7521, '2022-02-10 14:30:59', '广东', '梅州', 18, 0, 16, 0);
INSERT INTO `details` VALUES (7522, '2022-02-10 14:30:59', '广东', '河源', 6, 0, 5, 0);
INSERT INTO `details` VALUES (7523, '2022-02-10 14:30:59', '广东', '佛山', 101, 0, 101, 0);
INSERT INTO `details` VALUES (7524, '2022-02-10 14:30:59', '广东', '潮州', 5, 0, 5, 0);
INSERT INTO `details` VALUES (7525, '2022-02-10 14:30:59', '广东', '湛江', 23, 0, 23, 0);
INSERT INTO `details` VALUES (7526, '2022-02-10 14:30:59', '广东', '汕尾', 6, 0, 6, 0);
INSERT INTO `details` VALUES (7527, '2022-02-10 14:30:59', '广东', '汕头', 25, 0, 25, 0);
INSERT INTO `details` VALUES (7528, '2022-02-10 14:30:59', '广东', '揭阳', 10, 0, 10, 0);
INSERT INTO `details` VALUES (7529, '2022-02-10 14:30:59', '广东', '江门', 23, 0, 23, 0);
INSERT INTO `details` VALUES (7530, '2022-02-10 14:30:59', '广东', '阳江', 14, 0, 14, 0);
INSERT INTO `details` VALUES (7531, '2022-02-10 14:30:59', '广东', '茂名', 14, 0, 14, 0);
INSERT INTO `details` VALUES (7532, '2022-02-10 14:30:59', '广东', '韶关', 10, 0, 9, 1);
INSERT INTO `details` VALUES (7533, '2022-02-10 14:30:59', '广东', '清远', 12, 0, 12, 0);
INSERT INTO `details` VALUES (7534, '2022-02-10 14:30:59', '广东', '肇庆', 19, 0, 18, 1);
INSERT INTO `details` VALUES (7535, '2022-02-10 14:30:59', '广东', '地区待确认', 0, 0, 85, 0);
INSERT INTO `details` VALUES (7536, '2022-02-10 14:30:59', '北京', '丰台', 347, 0, 237, 0);
INSERT INTO `details` VALUES (7537, '2022-02-10 14:30:59', '北京', '朝阳', 96, 0, 9, 0);
INSERT INTO `details` VALUES (7538, '2022-02-10 14:30:59', '北京', '海淀', 96, 0, 26, 0);
INSERT INTO `details` VALUES (7539, '2022-02-10 14:30:59', '北京', '西城', 63, 0, 6, 0);
INSERT INTO `details` VALUES (7540, '2022-02-10 14:30:59', '北京', '境外输入', 371, 0, 329, 0);
INSERT INTO `details` VALUES (7541, '2022-02-10 14:30:59', '北京', '大兴', 144, 0, 111, 0);
INSERT INTO `details` VALUES (7542, '2022-02-10 14:30:59', '北京', '外地来京', 25, 0, 2, 0);
INSERT INTO `details` VALUES (7543, '2022-02-10 14:30:59', '北京', '房山', 31, 0, 12, 0);
INSERT INTO `details` VALUES (7544, '2022-02-10 14:30:59', '北京', '昌平', 69, 0, 50, 0);
INSERT INTO `details` VALUES (7545, '2022-02-10 14:30:59', '北京', '东城', 19, 0, 6, 0);
INSERT INTO `details` VALUES (7546, '2022-02-10 14:30:59', '北京', '通州', 21, 0, 3, 9);
INSERT INTO `details` VALUES (7547, '2022-02-10 14:30:59', '北京', '密云', 7, 0, 0, 0);
INSERT INTO `details` VALUES (7548, '2022-02-10 14:30:59', '北京', '怀柔', 8, 0, 4, 0);
INSERT INTO `details` VALUES (7549, '2022-02-10 14:30:59', '北京', '石景山', 15, 0, 12, 0);
INSERT INTO `details` VALUES (7550, '2022-02-10 14:30:59', '北京', '门头沟', 5, 0, 4, 0);
INSERT INTO `details` VALUES (7551, '2022-02-10 14:30:59', '北京', '延庆', 1, 0, 0, 0);
INSERT INTO `details` VALUES (7552, '2022-02-10 14:30:59', '北京', '顺义', 45, 0, 45, 0);
INSERT INTO `details` VALUES (7553, '2022-02-10 14:30:59', '北京', '地区待确认', 2, 0, 409, 0);
INSERT INTO `details` VALUES (7554, '2022-02-10 14:30:59', '天津', '河北区', 56, 0, 12, 0);
INSERT INTO `details` VALUES (7555, '2022-02-10 14:30:59', '天津', '津南区', 344, 0, 306, 0);
INSERT INTO `details` VALUES (7556, '2022-02-10 14:30:59', '天津', '西青区', 17, 0, 5, 0);
INSERT INTO `details` VALUES (7557, '2022-02-10 14:30:59', '天津', '滨海新区', 22, 0, 13, 0);
INSERT INTO `details` VALUES (7558, '2022-02-10 14:30:59', '天津', '河西区', 10, 0, 4, 0);
INSERT INTO `details` VALUES (7559, '2022-02-10 14:30:59', '天津', '红桥区', 7, 0, 2, 0);
INSERT INTO `details` VALUES (7560, '2022-02-10 14:30:59', '天津', '河东区', 19, 0, 14, 1);
INSERT INTO `details` VALUES (7561, '2022-02-10 14:30:59', '天津', '境外输入', 513, 0, 509, 0);
INSERT INTO `details` VALUES (7562, '2022-02-10 14:30:59', '天津', '东丽区', 5, 0, 4, 0);
INSERT INTO `details` VALUES (7563, '2022-02-10 14:30:59', '天津', '南开区', 7, 0, 7, 0);
INSERT INTO `details` VALUES (7564, '2022-02-10 14:30:59', '天津', '外地来津', 6, 0, 6, 0);
INSERT INTO `details` VALUES (7565, '2022-02-10 14:30:59', '天津', '北辰区', 7, 0, 7, 0);
INSERT INTO `details` VALUES (7566, '2022-02-10 14:30:59', '天津', '宝坻区', 60, 0, 58, 2);
INSERT INTO `details` VALUES (7567, '2022-02-10 14:30:59', '天津', '宁河区', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7568, '2022-02-10 14:30:59', '天津', '和平区', 6, 0, 6, 0);
INSERT INTO `details` VALUES (7569, '2022-02-10 14:30:59', '天津', '武清区', 2, 0, 2, 0);
INSERT INTO `details` VALUES (7570, '2022-02-10 14:30:59', '天津', '待确认', 0, 0, 84, 0);
INSERT INTO `details` VALUES (7571, '2022-02-10 14:30:59', '云南', '境外输入', 1411, 0, 1380, 0);
INSERT INTO `details` VALUES (7572, '2022-02-10 14:30:59', '云南', '德宏州', 294, 0, 288, 0);
INSERT INTO `details` VALUES (7573, '2022-02-10 14:30:59', '云南', '昆明', 58, 0, 53, 0);
INSERT INTO `details` VALUES (7574, '2022-02-10 14:30:59', '云南', '西双版纳州', 20, 0, 14, 1);
INSERT INTO `details` VALUES (7575, '2022-02-10 14:30:59', '云南', '红河', 9, 0, 9, 0);
INSERT INTO `details` VALUES (7576, '2022-02-10 14:30:59', '云南', '曲靖', 13, 0, 13, 0);
INSERT INTO `details` VALUES (7577, '2022-02-10 14:30:59', '云南', '保山市', 9, 0, 9, 0);
INSERT INTO `details` VALUES (7578, '2022-02-10 14:30:59', '云南', '玉溪', 14, 0, 13, 1);
INSERT INTO `details` VALUES (7579, '2022-02-10 14:30:59', '云南', '丽江市', 7, 0, 7, 0);
INSERT INTO `details` VALUES (7580, '2022-02-10 14:30:59', '云南', '昭通市', 25, 0, 25, 0);
INSERT INTO `details` VALUES (7581, '2022-02-10 14:30:59', '云南', '大理', 13, 0, 13, 0);
INSERT INTO `details` VALUES (7582, '2022-02-10 14:30:59', '云南', '文山州', 2, 0, 2, 0);
INSERT INTO `details` VALUES (7583, '2022-02-10 14:30:59', '云南', '楚雄州', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7584, '2022-02-10 14:30:59', '云南', '普洱', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7585, '2022-02-10 14:30:59', '云南', '临沧', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7586, '2022-02-10 14:30:59', '云南', '地区待确认', 0, 0, 8, 0);
INSERT INTO `details` VALUES (7587, '2022-02-10 14:30:59', '福建', '境外输入', 694, 1, 660, 0);
INSERT INTO `details` VALUES (7588, '2022-02-10 14:30:59', '福建', '厦门', 276, 0, 276, 0);
INSERT INTO `details` VALUES (7589, '2022-02-10 14:30:59', '福建', '漳州', 23, 0, 23, 0);
INSERT INTO `details` VALUES (7590, '2022-02-10 14:30:59', '福建', '地区待确认', 0, 0, 0, 0);
INSERT INTO `details` VALUES (7591, '2022-02-10 14:30:59', '福建', '莆田', 261, 0, 261, 0);
INSERT INTO `details` VALUES (7592, '2022-02-10 14:30:59', '福建', '泉州', 71, 0, 71, 0);
INSERT INTO `details` VALUES (7593, '2022-02-10 14:30:59', '福建', '宁德', 26, 0, 26, 0);
INSERT INTO `details` VALUES (7594, '2022-02-10 14:30:59', '福建', '龙岩', 6, 0, 6, 0);
INSERT INTO `details` VALUES (7595, '2022-02-10 14:30:59', '福建', '南平', 20, 0, 20, 0);
INSERT INTO `details` VALUES (7596, '2022-02-10 14:30:59', '福建', '三明', 14, 0, 14, 0);
INSERT INTO `details` VALUES (7597, '2022-02-10 14:30:59', '福建', '福州', 72, 0, 71, 1);
INSERT INTO `details` VALUES (7598, '2022-02-10 14:30:59', '黑龙江', '牡丹江', 65, 0, 39, 0);
INSERT INTO `details` VALUES (7599, '2022-02-10 14:30:59', '黑龙江', '大庆', 29, 0, 28, 1);
INSERT INTO `details` VALUES (7600, '2022-02-10 14:30:59', '黑龙江', '齐齐哈尔', 45, 0, 44, 1);
INSERT INTO `details` VALUES (7601, '2022-02-10 14:30:59', '黑龙江', '黑河', 295, 0, 295, 0);
INSERT INTO `details` VALUES (7602, '2022-02-10 14:30:59', '黑龙江', '境外输入', 406, 0, 406, 0);
INSERT INTO `details` VALUES (7603, '2022-02-10 14:30:59', '黑龙江', '绥化', 537, 0, 533, 4);
INSERT INTO `details` VALUES (7604, '2022-02-10 14:30:59', '黑龙江', '地区待确认', 2, 0, 2, 0);
INSERT INTO `details` VALUES (7605, '2022-02-10 14:30:59', '黑龙江', '哈尔滨', 546, 0, 542, 4);
INSERT INTO `details` VALUES (7606, '2022-02-10 14:30:59', '黑龙江', '大兴安岭', 3, 0, 3, 0);
INSERT INTO `details` VALUES (7607, '2022-02-10 14:30:59', '黑龙江', '双鸭山', 52, 0, 49, 3);
INSERT INTO `details` VALUES (7608, '2022-02-10 14:30:59', '黑龙江', '鹤岗', 5, 0, 5, 0);
INSERT INTO `details` VALUES (7609, '2022-02-10 14:30:59', '黑龙江', '七台河', 17, 0, 17, 0);
INSERT INTO `details` VALUES (7610, '2022-02-10 14:30:59', '黑龙江', '鸡西', 46, 0, 46, 0);
INSERT INTO `details` VALUES (7611, '2022-02-10 14:30:59', '黑龙江', '伊春', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7612, '2022-02-10 14:30:59', '黑龙江', '佳木斯', 15, 0, 15, 0);
INSERT INTO `details` VALUES (7613, '2022-02-10 14:30:59', '四川', '境外输入', 790, 0, 763, 0);
INSERT INTO `details` VALUES (7614, '2022-02-10 14:30:59', '四川', '成都', 191, 0, 174, 3);
INSERT INTO `details` VALUES (7615, '2022-02-10 14:30:59', '四川', '遂宁', 17, 0, 17, 0);
INSERT INTO `details` VALUES (7616, '2022-02-10 14:30:59', '四川', '自贡', 10, 0, 10, 0);
INSERT INTO `details` VALUES (7617, '2022-02-10 14:30:59', '四川', '宜宾', 13, 0, 13, 0);
INSERT INTO `details` VALUES (7618, '2022-02-10 14:30:59', '四川', '泸州', 25, 0, 25, 0);
INSERT INTO `details` VALUES (7619, '2022-02-10 14:30:59', '四川', '绵阳', 23, 0, 23, 0);
INSERT INTO `details` VALUES (7620, '2022-02-10 14:30:59', '四川', '雅安', 8, 0, 8, 0);
INSERT INTO `details` VALUES (7621, '2022-02-10 14:30:59', '四川', '内江', 22, 0, 22, 0);
INSERT INTO `details` VALUES (7622, '2022-02-10 14:30:59', '四川', '乐山', 3, 0, 3, 0);
INSERT INTO `details` VALUES (7623, '2022-02-10 14:30:59', '四川', '攀枝花', 16, 0, 16, 0);
INSERT INTO `details` VALUES (7624, '2022-02-10 14:30:59', '四川', '德阳', 18, 0, 18, 0);
INSERT INTO `details` VALUES (7625, '2022-02-10 14:30:59', '四川', '广元', 6, 0, 6, 0);
INSERT INTO `details` VALUES (7626, '2022-02-10 14:30:59', '四川', '凉山', 13, 0, 13, 0);
INSERT INTO `details` VALUES (7627, '2022-02-10 14:30:59', '四川', '甘孜', 78, 0, 78, 0);
INSERT INTO `details` VALUES (7628, '2022-02-10 14:30:59', '四川', '阿坝', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7629, '2022-02-10 14:30:59', '四川', '资阳', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7630, '2022-02-10 14:30:59', '四川', '眉山', 8, 0, 8, 0);
INSERT INTO `details` VALUES (7631, '2022-02-10 14:30:59', '四川', '巴中', 24, 0, 24, 0);
INSERT INTO `details` VALUES (7632, '2022-02-10 14:30:59', '四川', '达州', 42, 0, 42, 0);
INSERT INTO `details` VALUES (7633, '2022-02-10 14:30:59', '四川', '广安', 30, 0, 30, 0);
INSERT INTO `details` VALUES (7634, '2022-02-10 14:30:59', '四川', '南充', 39, 0, 39, 0);
INSERT INTO `details` VALUES (7635, '2022-02-10 14:30:59', '四川', '地区待确认', 0, 0, 22, 0);
INSERT INTO `details` VALUES (7636, '2022-02-10 14:30:59', '山东', '境外输入', 270, 0, 254, 0);
INSERT INTO `details` VALUES (7637, '2022-02-10 14:30:59', '山东', '济南', 49, 0, 47, 0);
INSERT INTO `details` VALUES (7638, '2022-02-10 14:30:59', '山东', '威海', 38, 0, 37, 1);
INSERT INTO `details` VALUES (7639, '2022-02-10 14:30:59', '山东', '烟台', 58, 0, 58, 0);
INSERT INTO `details` VALUES (7640, '2022-02-10 14:30:59', '山东', '青岛', 80, 0, 79, 1);
INSERT INTO `details` VALUES (7641, '2022-02-10 14:30:59', '山东', '济宁', 260, 0, 260, 0);
INSERT INTO `details` VALUES (7642, '2022-02-10 14:30:59', '山东', '淄博', 30, 0, 29, 1);
INSERT INTO `details` VALUES (7643, '2022-02-10 14:30:59', '山东', '泰安', 35, 0, 33, 2);
INSERT INTO `details` VALUES (7644, '2022-02-10 14:30:59', '山东', '日照', 30, 0, 30, 0);
INSERT INTO `details` VALUES (7645, '2022-02-10 14:30:59', '山东', '临沂', 49, 0, 49, 0);
INSERT INTO `details` VALUES (7646, '2022-02-10 14:30:59', '山东', '德州', 37, 0, 35, 2);
INSERT INTO `details` VALUES (7647, '2022-02-10 14:30:59', '山东', '聊城', 38, 0, 38, 0);
INSERT INTO `details` VALUES (7648, '2022-02-10 14:30:59', '山东', '滨州', 15, 0, 15, 0);
INSERT INTO `details` VALUES (7649, '2022-02-10 14:30:59', '山东', '菏泽', 18, 0, 18, 0);
INSERT INTO `details` VALUES (7650, '2022-02-10 14:30:59', '山东', '枣庄', 24, 0, 24, 0);
INSERT INTO `details` VALUES (7651, '2022-02-10 14:30:59', '山东', '潍坊', 44, 0, 44, 0);
INSERT INTO `details` VALUES (7652, '2022-02-10 14:30:59', '河北', '廊坊', 39, 0, 33, 0);
INSERT INTO `details` VALUES (7653, '2022-02-10 14:30:59', '河北', '衡水', 13, 0, 8, 0);
INSERT INTO `details` VALUES (7654, '2022-02-10 14:30:59', '河北', '雄安新区', 18, 0, 13, 0);
INSERT INTO `details` VALUES (7655, '2022-02-10 14:30:59', '河北', '保定', 38, 0, 36, 0);
INSERT INTO `details` VALUES (7656, '2022-02-10 14:30:59', '河北', '邢台', 96, 0, 95, 1);
INSERT INTO `details` VALUES (7657, '2022-02-10 14:30:59', '河北', '石家庄', 963, 0, 962, 1);
INSERT INTO `details` VALUES (7658, '2022-02-10 14:30:59', '河北', '地区待确认', 0, 0, 0, 0);
INSERT INTO `details` VALUES (7659, '2022-02-10 14:30:59', '河北', '辛集市', 71, 0, 71, 0);
INSERT INTO `details` VALUES (7660, '2022-02-10 14:30:59', '河北', '定州', 2, 0, 2, 0);
INSERT INTO `details` VALUES (7661, '2022-02-10 14:30:59', '河北', '境外输入', 37, 0, 37, 0);
INSERT INTO `details` VALUES (7662, '2022-02-10 14:30:59', '河北', '沧州', 49, 0, 46, 3);
INSERT INTO `details` VALUES (7663, '2022-02-10 14:30:59', '河北', '张家口', 43, 0, 43, 0);
INSERT INTO `details` VALUES (7664, '2022-02-10 14:30:59', '河北', '唐山', 58, 0, 57, 1);
INSERT INTO `details` VALUES (7665, '2022-02-10 14:30:59', '河北', '邯郸', 32, 0, 32, 0);
INSERT INTO `details` VALUES (7666, '2022-02-10 14:30:59', '河北', '秦皇岛', 10, 0, 9, 1);
INSERT INTO `details` VALUES (7667, '2022-02-10 14:30:59', '河北', '承德', 7, 0, 7, 0);
INSERT INTO `details` VALUES (7668, '2022-02-10 14:30:59', '湖南', '境外输入', 98, 0, 92, 0);
INSERT INTO `details` VALUES (7669, '2022-02-10 14:30:59', '湖南', '邵阳', 103, 0, 101, 1);
INSERT INTO `details` VALUES (7670, '2022-02-10 14:30:59', '湖南', '益阳', 63, 0, 63, 0);
INSERT INTO `details` VALUES (7671, '2022-02-10 14:30:59', '湖南', '长沙', 247, 0, 245, 2);
INSERT INTO `details` VALUES (7672, '2022-02-10 14:30:59', '湖南', '地区待确认', 0, 0, 0, 0);
INSERT INTO `details` VALUES (7673, '2022-02-10 14:30:59', '湖南', '湘潭', 38, 0, 38, 0);
INSERT INTO `details` VALUES (7674, '2022-02-10 14:30:59', '湖南', '湘西自治州', 10, 0, 10, 0);
INSERT INTO `details` VALUES (7675, '2022-02-10 14:30:59', '湖南', '张家界', 77, 0, 77, 0);
INSERT INTO `details` VALUES (7676, '2022-02-10 14:30:59', '湖南', '株洲', 110, 0, 110, 0);
INSERT INTO `details` VALUES (7677, '2022-02-10 14:30:59', '湖南', '娄底', 76, 0, 76, 0);
INSERT INTO `details` VALUES (7678, '2022-02-10 14:30:59', '湖南', '怀化', 40, 0, 40, 0);
INSERT INTO `details` VALUES (7679, '2022-02-10 14:30:59', '湖南', '永州', 44, 0, 44, 0);
INSERT INTO `details` VALUES (7680, '2022-02-10 14:30:59', '湖南', '郴州', 39, 0, 39, 0);
INSERT INTO `details` VALUES (7681, '2022-02-10 14:30:59', '湖南', '常德', 82, 0, 82, 0);
INSERT INTO `details` VALUES (7682, '2022-02-10 14:30:59', '湖南', '岳阳', 156, 0, 155, 1);
INSERT INTO `details` VALUES (7683, '2022-02-10 14:30:59', '湖南', '衡阳', 48, 0, 48, 0);
INSERT INTO `details` VALUES (7684, '2022-02-10 14:30:59', '吉林', '境外输入', 38, 2, 33, 0);
INSERT INTO `details` VALUES (7685, '2022-02-10 14:30:59', '吉林', '吉林', 50, 0, 49, 1);
INSERT INTO `details` VALUES (7686, '2022-02-10 14:30:59', '吉林', '通化', 313, 0, 312, 1);
INSERT INTO `details` VALUES (7687, '2022-02-10 14:30:59', '吉林', '长春', 150, 0, 150, 0);
INSERT INTO `details` VALUES (7688, '2022-02-10 14:30:59', '吉林', '松原', 6, 0, 6, 0);
INSERT INTO `details` VALUES (7689, '2022-02-10 14:30:59', '吉林', '四平', 17, 0, 16, 1);
INSERT INTO `details` VALUES (7690, '2022-02-10 14:30:59', '吉林', '白城', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7691, '2022-02-10 14:30:59', '吉林', '公主岭', 6, 0, 6, 0);
INSERT INTO `details` VALUES (7692, '2022-02-10 14:30:59', '吉林', '辽源', 7, 0, 7, 0);
INSERT INTO `details` VALUES (7693, '2022-02-10 14:30:59', '吉林', '延边', 5, 0, 5, 0);
INSERT INTO `details` VALUES (7694, '2022-02-10 14:30:59', '吉林', '梅河口市', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7695, '2022-02-10 14:30:59', '新疆', '兵团第四师', 13, 0, 9, 1);
INSERT INTO `details` VALUES (7696, '2022-02-10 14:30:59', '新疆', '伊犁州', 31, 0, 29, 0);
INSERT INTO `details` VALUES (7697, '2022-02-10 14:30:59', '新疆', '吐鲁番', 3, 0, 3, 0);
INSERT INTO `details` VALUES (7698, '2022-02-10 14:30:59', '新疆', '乌鲁木齐', 845, 0, 845, 0);
INSERT INTO `details` VALUES (7699, '2022-02-10 14:30:59', '新疆', '地区待确认', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7700, '2022-02-10 14:30:59', '新疆', '昌吉州', 5, 0, 5, 0);
INSERT INTO `details` VALUES (7701, '2022-02-10 14:30:59', '新疆', '兵团第九师', 4, 0, 3, 1);
INSERT INTO `details` VALUES (7702, '2022-02-10 14:30:59', '新疆', '喀什', 80, 0, 80, 0);
INSERT INTO `details` VALUES (7703, '2022-02-10 14:30:59', '新疆', '第八师石河子', 4, 0, 3, 1);
INSERT INTO `details` VALUES (7704, '2022-02-10 14:30:59', '新疆', '六师五家渠', 2, 0, 2, 0);
INSERT INTO `details` VALUES (7705, '2022-02-10 14:30:59', '新疆', '兵团第十二师', 3, 0, 3, 0);
INSERT INTO `details` VALUES (7706, '2022-02-10 14:30:59', '新疆', '巴州', 3, 0, 3, 0);
INSERT INTO `details` VALUES (7707, '2022-02-10 14:30:59', '新疆', '第七师', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7708, '2022-02-10 14:30:59', '新疆', '阿克苏', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7709, '2022-02-10 14:30:59', '辽宁', '境外输入', 173, 0, 171, 0);
INSERT INTO `details` VALUES (7710, '2022-02-10 14:30:59', '辽宁', '葫芦岛', 13, 0, 11, 1);
INSERT INTO `details` VALUES (7711, '2022-02-10 14:30:59', '辽宁', '沈阳', 79, 0, 78, 0);
INSERT INTO `details` VALUES (7712, '2022-02-10 14:30:59', '辽宁', '朝阳市', 6, 0, 5, 1);
INSERT INTO `details` VALUES (7713, '2022-02-10 14:30:59', '辽宁', '地区待确认', 0, 0, 0, 0);
INSERT INTO `details` VALUES (7714, '2022-02-10 14:30:59', '辽宁', '营口', 11, 0, 11, 0);
INSERT INTO `details` VALUES (7715, '2022-02-10 14:30:59', '辽宁', '铁岭', 8, 0, 8, 0);
INSERT INTO `details` VALUES (7716, '2022-02-10 14:30:59', '辽宁', '抚顺', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7717, '2022-02-10 14:30:59', '辽宁', '大连', 470, 0, 470, 0);
INSERT INTO `details` VALUES (7718, '2022-02-10 14:30:59', '辽宁', '丹东', 11, 0, 11, 0);
INSERT INTO `details` VALUES (7719, '2022-02-10 14:30:59', '辽宁', '鞍山', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7720, '2022-02-10 14:30:59', '辽宁', '锦州', 12, 0, 12, 0);
INSERT INTO `details` VALUES (7721, '2022-02-10 14:30:59', '辽宁', '盘锦', 11, 0, 11, 0);
INSERT INTO `details` VALUES (7722, '2022-02-10 14:30:59', '辽宁', '阜新', 8, 0, 8, 0);
INSERT INTO `details` VALUES (7723, '2022-02-10 14:30:59', '辽宁', '本溪', 3, 0, 3, 0);
INSERT INTO `details` VALUES (7724, '2022-02-10 14:30:59', '辽宁', '辽阳', 3, 0, 3, 0);
INSERT INTO `details` VALUES (7725, '2022-02-10 14:30:59', '山西', '大同', 14, 0, 12, 0);
INSERT INTO `details` VALUES (7726, '2022-02-10 14:30:59', '山西', '运城', 20, 0, 20, 0);
INSERT INTO `details` VALUES (7727, '2022-02-10 14:30:59', '山西', '境外输入', 127, 0, 127, 0);
INSERT INTO `details` VALUES (7728, '2022-02-10 14:30:59', '山西', '晋中', 41, 0, 41, 0);
INSERT INTO `details` VALUES (7729, '2022-02-10 14:30:59', '山西', '太原', 21, 0, 21, 0);
INSERT INTO `details` VALUES (7730, '2022-02-10 14:30:59', '山西', '晋城', 10, 0, 10, 0);
INSERT INTO `details` VALUES (7731, '2022-02-10 14:30:59', '山西', '忻州', 7, 0, 7, 0);
INSERT INTO `details` VALUES (7732, '2022-02-10 14:30:59', '山西', '长治', 8, 0, 8, 0);
INSERT INTO `details` VALUES (7733, '2022-02-10 14:30:59', '山西', '朔州', 8, 0, 8, 0);
INSERT INTO `details` VALUES (7734, '2022-02-10 14:30:59', '山西', '临汾', 2, 0, 2, 0);
INSERT INTO `details` VALUES (7735, '2022-02-10 14:30:59', '山西', '阳泉', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7736, '2022-02-10 14:30:59', '山西', '吕梁', 6, 0, 6, 0);
INSERT INTO `details` VALUES (7737, '2022-02-10 14:30:59', '重庆', '境外输入', 29, 0, 27, 0);
INSERT INTO `details` VALUES (7738, '2022-02-10 14:30:59', '重庆', '奉节县', 22, 0, 22, 0);
INSERT INTO `details` VALUES (7739, '2022-02-10 14:30:59', '重庆', '巴南区', 7, 0, 7, 0);
INSERT INTO `details` VALUES (7740, '2022-02-10 14:30:59', '重庆', '两江新区', 18, 0, 18, 0);
INSERT INTO `details` VALUES (7741, '2022-02-10 14:30:59', '重庆', '九龙坡区', 22, 0, 21, 1);
INSERT INTO `details` VALUES (7742, '2022-02-10 14:30:59', '重庆', '高新区', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7743, '2022-02-10 14:30:59', '重庆', '沙坪坝区', 10, 0, 10, 0);
INSERT INTO `details` VALUES (7744, '2022-02-10 14:30:59', '重庆', '江津区', 6, 0, 6, 0);
INSERT INTO `details` VALUES (7745, '2022-02-10 14:30:59', '重庆', '綦江区', 23, 0, 23, 0);
INSERT INTO `details` VALUES (7746, '2022-02-10 14:30:59', '重庆', '大足区', 14, 0, 14, 0);
INSERT INTO `details` VALUES (7747, '2022-02-10 14:30:59', '重庆', '荣昌区', 9, 0, 9, 0);
INSERT INTO `details` VALUES (7748, '2022-02-10 14:30:59', '重庆', '江北区', 28, 0, 28, 0);
INSERT INTO `details` VALUES (7749, '2022-02-10 14:30:59', '重庆', '丰都县', 10, 0, 10, 0);
INSERT INTO `details` VALUES (7750, '2022-02-10 14:30:59', '重庆', '潼南区', 18, 0, 18, 0);
INSERT INTO `details` VALUES (7751, '2022-02-10 14:30:59', '重庆', '铜梁区', 10, 0, 10, 0);
INSERT INTO `details` VALUES (7752, '2022-02-10 14:30:59', '重庆', '南岸区', 15, 0, 15, 0);
INSERT INTO `details` VALUES (7753, '2022-02-10 14:30:59', '重庆', '万州区', 118, 0, 114, 4);
INSERT INTO `details` VALUES (7754, '2022-02-10 14:30:59', '重庆', '渝中区', 20, 0, 20, 0);
INSERT INTO `details` VALUES (7755, '2022-02-10 14:30:59', '重庆', '垫江县', 20, 0, 20, 0);
INSERT INTO `details` VALUES (7756, '2022-02-10 14:30:59', '重庆', '云阳县', 25, 0, 25, 0);
INSERT INTO `details` VALUES (7757, '2022-02-10 14:30:59', '重庆', '长寿区', 24, 0, 24, 0);
INSERT INTO `details` VALUES (7758, '2022-02-10 14:30:59', '重庆', '石柱县', 15, 0, 15, 0);
INSERT INTO `details` VALUES (7759, '2022-02-10 14:30:59', '重庆', '涪陵区', 5, 0, 5, 0);
INSERT INTO `details` VALUES (7760, '2022-02-10 14:30:59', '重庆', '渝北区', 17, 0, 17, 0);
INSERT INTO `details` VALUES (7761, '2022-02-10 14:30:59', '重庆', '彭水县', 2, 0, 2, 0);
INSERT INTO `details` VALUES (7762, '2022-02-10 14:30:59', '重庆', '忠县', 21, 0, 21, 0);
INSERT INTO `details` VALUES (7763, '2022-02-10 14:30:59', '重庆', '合川区', 23, 0, 23, 0);
INSERT INTO `details` VALUES (7764, '2022-02-10 14:30:59', '重庆', '开州区', 21, 0, 20, 1);
INSERT INTO `details` VALUES (7765, '2022-02-10 14:30:59', '重庆', '巫溪县', 14, 0, 14, 0);
INSERT INTO `details` VALUES (7766, '2022-02-10 14:30:59', '重庆', '大渡口区', 7, 0, 7, 0);
INSERT INTO `details` VALUES (7767, '2022-02-10 14:30:59', '重庆', '巫山县', 10, 0, 10, 0);
INSERT INTO `details` VALUES (7768, '2022-02-10 14:30:59', '重庆', '万盛经开区', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7769, '2022-02-10 14:30:59', '重庆', '酉阳县', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7770, '2022-02-10 14:30:59', '重庆', '璧山区', 9, 0, 9, 0);
INSERT INTO `details` VALUES (7771, '2022-02-10 14:30:59', '重庆', '永川区', 5, 0, 5, 0);
INSERT INTO `details` VALUES (7772, '2022-02-10 14:30:59', '重庆', '武隆区', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7773, '2022-02-10 14:30:59', '重庆', '梁平区', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7774, '2022-02-10 14:30:59', '重庆', '城口县', 2, 0, 2, 0);
INSERT INTO `details` VALUES (7775, '2022-02-10 14:30:59', '重庆', '黔江区', 2, 0, 2, 0);
INSERT INTO `details` VALUES (7776, '2022-02-10 14:30:59', '重庆', '秀山县', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7777, '2022-02-10 14:30:59', '江苏', '境外输入', 174, 0, 172, 0);
INSERT INTO `details` VALUES (7778, '2022-02-10 14:30:59', '江苏', '淮安', 78, 0, 78, 0);
INSERT INTO `details` VALUES (7779, '2022-02-10 14:30:59', '江苏', '无锡', 55, 0, 55, 0);
INSERT INTO `details` VALUES (7780, '2022-02-10 14:30:59', '江苏', '常州', 54, 0, 54, 0);
INSERT INTO `details` VALUES (7781, '2022-02-10 14:30:59', '江苏', '地区待确认', 0, 0, 0, 0);
INSERT INTO `details` VALUES (7782, '2022-02-10 14:30:59', '江苏', '扬州', 593, 0, 593, 0);
INSERT INTO `details` VALUES (7783, '2022-02-10 14:30:59', '江苏', '宿迁', 16, 0, 16, 0);
INSERT INTO `details` VALUES (7784, '2022-02-10 14:30:59', '江苏', '南京', 331, 0, 331, 0);
INSERT INTO `details` VALUES (7785, '2022-02-10 14:30:59', '江苏', '盐城', 27, 0, 27, 0);
INSERT INTO `details` VALUES (7786, '2022-02-10 14:30:59', '江苏', '南通', 40, 0, 40, 0);
INSERT INTO `details` VALUES (7787, '2022-02-10 14:30:59', '江苏', '苏州', 87, 0, 87, 0);
INSERT INTO `details` VALUES (7788, '2022-02-10 14:30:59', '江苏', '徐州', 79, 0, 79, 0);
INSERT INTO `details` VALUES (7789, '2022-02-10 14:30:59', '江苏', '连云港', 48, 0, 48, 0);
INSERT INTO `details` VALUES (7790, '2022-02-10 14:30:59', '江苏', '镇江', 12, 0, 12, 0);
INSERT INTO `details` VALUES (7791, '2022-02-10 14:30:59', '江苏', '泰州', 37, 0, 37, 0);
INSERT INTO `details` VALUES (7792, '2022-02-10 14:30:59', '贵州', '安顺', 5, 0, 5, 0);
INSERT INTO `details` VALUES (7793, '2022-02-10 14:30:59', '贵州', '铜仁', 11, 0, 11, 0);
INSERT INTO `details` VALUES (7794, '2022-02-10 14:30:59', '贵州', '遵义', 44, 0, 44, 0);
INSERT INTO `details` VALUES (7795, '2022-02-10 14:30:59', '贵州', '境外输入', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7796, '2022-02-10 14:30:59', '贵州', '贵阳', 36, 0, 35, 1);
INSERT INTO `details` VALUES (7797, '2022-02-10 14:30:59', '贵州', '六盘水', 10, 0, 9, 1);
INSERT INTO `details` VALUES (7798, '2022-02-10 14:30:59', '贵州', '毕节', 23, 0, 23, 0);
INSERT INTO `details` VALUES (7799, '2022-02-10 14:30:59', '贵州', '黔南州', 17, 0, 17, 0);
INSERT INTO `details` VALUES (7800, '2022-02-10 14:30:59', '贵州', '黔东南州', 10, 0, 10, 0);
INSERT INTO `details` VALUES (7801, '2022-02-10 14:30:59', '贵州', '黔西南州', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7802, '2022-02-10 14:30:59', '内蒙古', '境外输入', 336, 0, 336, 0);
INSERT INTO `details` VALUES (7803, '2022-02-10 14:30:59', '内蒙古', '呼伦贝尔', 596, 0, 596, 0);
INSERT INTO `details` VALUES (7804, '2022-02-10 14:30:59', '内蒙古', '通辽', 9, 0, 9, 0);
INSERT INTO `details` VALUES (7805, '2022-02-10 14:30:59', '内蒙古', '阿拉善盟', 165, 0, 165, 0);
INSERT INTO `details` VALUES (7806, '2022-02-10 14:30:59', '内蒙古', '锡林郭勒', 29, 0, 29, 0);
INSERT INTO `details` VALUES (7807, '2022-02-10 14:30:59', '内蒙古', '鄂尔多斯', 12, 0, 12, 0);
INSERT INTO `details` VALUES (7808, '2022-02-10 14:30:59', '内蒙古', '呼和浩特', 10, 0, 10, 0);
INSERT INTO `details` VALUES (7809, '2022-02-10 14:30:59', '内蒙古', '包头', 11, 0, 11, 0);
INSERT INTO `details` VALUES (7810, '2022-02-10 14:30:59', '内蒙古', '乌兰察布', 3, 0, 3, 0);
INSERT INTO `details` VALUES (7811, '2022-02-10 14:30:59', '内蒙古', '赤峰', 9, 0, 9, 0);
INSERT INTO `details` VALUES (7812, '2022-02-10 14:30:59', '内蒙古', '巴彦淖尔', 8, 0, 7, 1);
INSERT INTO `details` VALUES (7813, '2022-02-10 14:30:59', '内蒙古', '兴安盟', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7814, '2022-02-10 14:30:59', '内蒙古', '乌海', 2, 0, 2, 0);
INSERT INTO `details` VALUES (7815, '2022-02-10 14:30:59', '陕西', '渭南', 18, 0, 18, 0);
INSERT INTO `details` VALUES (7816, '2022-02-10 14:30:59', '陕西', '延安', 21, 0, 21, 0);
INSERT INTO `details` VALUES (7817, '2022-02-10 14:30:59', '陕西', '咸阳', 31, 0, 31, 0);
INSERT INTO `details` VALUES (7818, '2022-02-10 14:30:59', '陕西', '西安', 2188, 0, 2185, 3);
INSERT INTO `details` VALUES (7819, '2022-02-10 14:30:59', '陕西', '地区待确认', 0, 0, 0, 0);
INSERT INTO `details` VALUES (7820, '2022-02-10 14:30:59', '陕西', '境外输入', 483, 0, 483, 0);
INSERT INTO `details` VALUES (7821, '2022-02-10 14:30:59', '陕西', '汉中', 26, 0, 26, 0);
INSERT INTO `details` VALUES (7822, '2022-02-10 14:30:59', '陕西', '杨凌', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7823, '2022-02-10 14:30:59', '陕西', '韩城', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7824, '2022-02-10 14:30:59', '陕西', '榆林', 3, 0, 3, 0);
INSERT INTO `details` VALUES (7825, '2022-02-10 14:30:59', '陕西', '商洛', 7, 0, 7, 0);
INSERT INTO `details` VALUES (7826, '2022-02-10 14:30:59', '陕西', '铜川', 8, 0, 8, 0);
INSERT INTO `details` VALUES (7827, '2022-02-10 14:30:59', '陕西', '宝鸡', 13, 0, 13, 0);
INSERT INTO `details` VALUES (7828, '2022-02-10 14:30:59', '陕西', '安康', 26, 0, 26, 0);
INSERT INTO `details` VALUES (7829, '2022-02-10 14:30:59', '澳门', '地区待确认', 79, 0, 79, 0);
INSERT INTO `details` VALUES (7830, '2022-02-10 14:30:59', '湖北', '境外输入', 92, 0, 92, 0);
INSERT INTO `details` VALUES (7831, '2022-02-10 14:30:59', '湖北', '天门', 498, 0, 483, 15);
INSERT INTO `details` VALUES (7832, '2022-02-10 14:30:59', '湖北', '荆门', 971, 0, 930, 41);
INSERT INTO `details` VALUES (7833, '2022-02-10 14:30:59', '湖北', '荆州', 1582, 0, 1530, 52);
INSERT INTO `details` VALUES (7834, '2022-02-10 14:30:59', '湖北', '武汉', 50380, 0, 46511, 3869);
INSERT INTO `details` VALUES (7835, '2022-02-10 14:30:59', '湖北', '黄冈', 2912, 0, 2787, 125);
INSERT INTO `details` VALUES (7836, '2022-02-10 14:30:59', '湖北', '鄂州', 1395, 0, 1336, 59);
INSERT INTO `details` VALUES (7837, '2022-02-10 14:30:59', '湖北', '宜昌', 931, 0, 894, 37);
INSERT INTO `details` VALUES (7838, '2022-02-10 14:30:59', '湖北', '十堰', 672, 0, 664, 8);
INSERT INTO `details` VALUES (7839, '2022-02-10 14:30:59', '湖北', '孝感', 3518, 0, 3389, 129);
INSERT INTO `details` VALUES (7840, '2022-02-10 14:30:59', '湖北', '仙桃', 575, 0, 553, 22);
INSERT INTO `details` VALUES (7841, '2022-02-10 14:30:59', '湖北', '襄阳', 1175, 0, 1135, 40);
INSERT INTO `details` VALUES (7842, '2022-02-10 14:30:59', '湖北', '潜江', 198, 0, 189, 9);
INSERT INTO `details` VALUES (7843, '2022-02-10 14:30:59', '湖北', '黄石', 1015, 0, 976, 39);
INSERT INTO `details` VALUES (7844, '2022-02-10 14:30:59', '湖北', '神农架', 11, 0, 11, 0);
INSERT INTO `details` VALUES (7845, '2022-02-10 14:30:59', '湖北', '随州', 1307, 0, 1262, 45);
INSERT INTO `details` VALUES (7846, '2022-02-10 14:30:59', '湖北', '咸宁', 836, 0, 821, 15);
INSERT INTO `details` VALUES (7847, '2022-02-10 14:30:59', '湖北', '恩施州', 252, 0, 245, 7);
INSERT INTO `details` VALUES (7848, '2022-02-10 14:30:59', '江西', '上饶', 144, 0, 144, 0);
INSERT INTO `details` VALUES (7849, '2022-02-10 14:30:59', '江西', '九江', 117, 0, 117, 0);
INSERT INTO `details` VALUES (7850, '2022-02-10 14:30:59', '江西', '境外输入', 6, 0, 6, 0);
INSERT INTO `details` VALUES (7851, '2022-02-10 14:30:59', '江西', '南昌', 231, 0, 231, 0);
INSERT INTO `details` VALUES (7852, '2022-02-10 14:30:59', '江西', '赣州', 74, 0, 73, 1);
INSERT INTO `details` VALUES (7853, '2022-02-10 14:30:59', '江西', '新余', 129, 0, 129, 0);
INSERT INTO `details` VALUES (7854, '2022-02-10 14:30:59', '江西', '抚州', 72, 0, 72, 0);
INSERT INTO `details` VALUES (7855, '2022-02-10 14:30:59', '江西', '吉安', 22, 0, 22, 0);
INSERT INTO `details` VALUES (7856, '2022-02-10 14:30:59', '江西', '萍乡', 33, 0, 33, 0);
INSERT INTO `details` VALUES (7857, '2022-02-10 14:30:59', '江西', '景德镇', 6, 0, 6, 0);
INSERT INTO `details` VALUES (7858, '2022-02-10 14:30:59', '江西', '鹰潭', 18, 0, 18, 0);
INSERT INTO `details` VALUES (7859, '2022-02-10 14:30:59', '江西', '宜春', 106, 0, 106, 0);
INSERT INTO `details` VALUES (7860, '2022-02-10 14:30:59', '江西', '赣江新区', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7861, '2022-02-10 14:30:59', '西藏', '拉萨', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7862, '2022-02-10 14:30:59', '青海', '西宁', 26, 0, 26, 0);
INSERT INTO `details` VALUES (7863, '2022-02-10 14:30:59', '青海', '海东', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7864, '2022-02-10 14:30:59', '青海', '海北州', 3, 0, 3, 0);
INSERT INTO `details` VALUES (7865, '2022-02-10 14:30:59', '甘肃', '境外输入', 120, 0, 120, 0);
INSERT INTO `details` VALUES (7866, '2022-02-10 14:30:59', '甘肃', '兰州', 119, 0, 117, 2);
INSERT INTO `details` VALUES (7867, '2022-02-10 14:30:59', '甘肃', '张掖', 17, 0, 17, 0);
INSERT INTO `details` VALUES (7868, '2022-02-10 14:30:59', '甘肃', '嘉峪关', 5, 0, 5, 0);
INSERT INTO `details` VALUES (7869, '2022-02-10 14:30:59', '甘肃', '天水', 51, 0, 51, 0);
INSERT INTO `details` VALUES (7870, '2022-02-10 14:30:59', '甘肃', '地区待确认', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7871, '2022-02-10 14:30:59', '甘肃', '陇南', 6, 0, 6, 0);
INSERT INTO `details` VALUES (7872, '2022-02-10 14:30:59', '甘肃', '定西', 9, 0, 9, 0);
INSERT INTO `details` VALUES (7873, '2022-02-10 14:30:59', '甘肃', '平凉', 9, 0, 9, 0);
INSERT INTO `details` VALUES (7874, '2022-02-10 14:30:59', '甘肃', '庆阳', 3, 0, 3, 0);
INSERT INTO `details` VALUES (7875, '2022-02-10 14:30:59', '甘肃', '白银', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7876, '2022-02-10 14:30:59', '甘肃', '甘南州', 8, 0, 8, 0);
INSERT INTO `details` VALUES (7877, '2022-02-10 14:30:59', '甘肃', '临夏', 3, 0, 3, 0);
INSERT INTO `details` VALUES (7878, '2022-02-10 14:30:59', '甘肃', '金昌', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7879, '2022-02-10 14:30:59', '安徽', '宿州', 42, 0, 42, 0);
INSERT INTO `details` VALUES (7880, '2022-02-10 14:30:59', '安徽', '境外输入', 7, 0, 7, 0);
INSERT INTO `details` VALUES (7881, '2022-02-10 14:30:59', '安徽', '六安', 77, 0, 77, 0);
INSERT INTO `details` VALUES (7882, '2022-02-10 14:30:59', '安徽', '合肥', 176, 0, 175, 1);
INSERT INTO `details` VALUES (7883, '2022-02-10 14:30:59', '安徽', '阜阳', 156, 0, 156, 0);
INSERT INTO `details` VALUES (7884, '2022-02-10 14:30:59', '安徽', '宣城', 6, 0, 6, 0);
INSERT INTO `details` VALUES (7885, '2022-02-10 14:30:59', '安徽', '池州', 17, 0, 17, 0);
INSERT INTO `details` VALUES (7886, '2022-02-10 14:30:59', '安徽', '亳州', 108, 0, 108, 0);
INSERT INTO `details` VALUES (7887, '2022-02-10 14:30:59', '安徽', '滁州', 13, 0, 13, 0);
INSERT INTO `details` VALUES (7888, '2022-02-10 14:30:59', '安徽', '黄山', 9, 0, 9, 0);
INSERT INTO `details` VALUES (7889, '2022-02-10 14:30:59', '安徽', '安庆', 83, 0, 83, 0);
INSERT INTO `details` VALUES (7890, '2022-02-10 14:30:59', '安徽', '铜陵', 29, 0, 29, 0);
INSERT INTO `details` VALUES (7891, '2022-02-10 14:30:59', '安徽', '淮北', 27, 0, 27, 0);
INSERT INTO `details` VALUES (7892, '2022-02-10 14:30:59', '安徽', '马鞍山', 38, 0, 38, 0);
INSERT INTO `details` VALUES (7893, '2022-02-10 14:30:59', '安徽', '淮南', 27, 0, 27, 0);
INSERT INTO `details` VALUES (7894, '2022-02-10 14:30:59', '安徽', '蚌埠', 160, 0, 155, 5);
INSERT INTO `details` VALUES (7895, '2022-02-10 14:30:59', '安徽', '芜湖', 34, 0, 34, 0);
INSERT INTO `details` VALUES (7896, '2022-02-10 14:30:59', '宁夏', '中卫', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7897, '2022-02-10 14:30:59', '宁夏', '地区待确认', 0, 0, 0, 0);
INSERT INTO `details` VALUES (7898, '2022-02-10 14:30:59', '宁夏', '银川', 68, 0, 68, 0);
INSERT INTO `details` VALUES (7899, '2022-02-10 14:30:59', '宁夏', '吴忠', 39, 0, 39, 0);
INSERT INTO `details` VALUES (7900, '2022-02-10 14:30:59', '宁夏', '境外输入', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7901, '2022-02-10 14:30:59', '宁夏', '固原', 5, 0, 5, 0);
INSERT INTO `details` VALUES (7902, '2022-02-10 14:30:59', '宁夏', '石嘴山', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7903, '2022-02-10 14:30:59', '宁夏', '宁东管委会', 1, 0, 1, 0);
INSERT INTO `details` VALUES (7904, '2022-02-10 14:30:59', '海南', '海口', 41, 0, 41, 0);
INSERT INTO `details` VALUES (7905, '2022-02-10 14:30:59', '海南', '境外输入', 19, 0, 19, 0);
INSERT INTO `details` VALUES (7906, '2022-02-10 14:30:59', '海南', '三亚', 55, 0, 54, 1);
INSERT INTO `details` VALUES (7907, '2022-02-10 14:30:59', '海南', '儋州', 15, 0, 14, 1);
INSERT INTO `details` VALUES (7908, '2022-02-10 14:30:59', '海南', '万宁', 13, 0, 13, 0);
INSERT INTO `details` VALUES (7909, '2022-02-10 14:30:59', '海南', '东方', 3, 0, 2, 1);
INSERT INTO `details` VALUES (7910, '2022-02-10 14:30:59', '海南', '澄迈县', 9, 0, 8, 1);
INSERT INTO `details` VALUES (7911, '2022-02-10 14:30:59', '海南', '昌江县', 7, 0, 7, 0);
INSERT INTO `details` VALUES (7912, '2022-02-10 14:30:59', '海南', '保亭', 3, 0, 3, 0);
INSERT INTO `details` VALUES (7913, '2022-02-10 14:30:59', '海南', '琼海', 6, 0, 5, 1);
INSERT INTO `details` VALUES (7914, '2022-02-10 14:30:59', '海南', '临高县', 6, 0, 6, 0);
INSERT INTO `details` VALUES (7915, '2022-02-10 14:30:59', '海南', '陵水县', 4, 0, 4, 0);
INSERT INTO `details` VALUES (7916, '2022-02-10 14:30:59', '海南', '乐东', 2, 0, 2, 0);
INSERT INTO `details` VALUES (7917, '2022-02-10 14:30:59', '海南', '文昌', 3, 0, 3, 0);
INSERT INTO `details` VALUES (7918, '2022-02-10 14:30:59', '海南', '定安县', 3, 0, 2, 1);
INSERT INTO `details` VALUES (7919, '2022-02-10 14:30:59', '海南', '琼中县', 1, 0, 1, 0);

-- ----------------------------
-- Table structure for history
-- ----------------------------
DROP TABLE IF EXISTS `history`;
CREATE TABLE `history`  (
  `ds` datetime NOT NULL COMMENT '日期',
  `confirm` int(11) NULL DEFAULT NULL COMMENT '累计确诊',
  `confirm_add` int(11) NULL DEFAULT NULL COMMENT '当日新增确诊',
  `suspect` int(11) NULL DEFAULT NULL COMMENT '剩余疑似',
  `suspect_add` int(11) NULL DEFAULT NULL COMMENT '当日新增疑似',
  `heal` int(11) NULL DEFAULT NULL COMMENT '累计治愈',
  `heal_add` int(11) NULL DEFAULT NULL COMMENT '当日新增治愈',
  `dead` int(11) NULL DEFAULT NULL COMMENT '累计死亡',
  `dead_add` int(11) NULL DEFAULT NULL COMMENT '当日新增死亡',
  PRIMARY KEY (`ds`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of history
-- ----------------------------
INSERT INTO `history` VALUES ('2021-12-12 00:00:00', 129082, 112, 2, 0, 119750, 48, 5697, 0);
INSERT INTO `history` VALUES ('2021-12-13 00:00:00', 129165, 83, 1, 0, 119781, 31, 5697, 0);
INSERT INTO `history` VALUES ('2021-12-14 00:00:00', 129247, 82, 1, 0, 119826, 45, 5697, 0);
INSERT INTO `history` VALUES ('2021-12-15 00:00:00', 129332, 86, 1, 0, 119859, 33, 5698, 1);
INSERT INTO `history` VALUES ('2021-12-16 00:00:00', 129430, 98, 4, 3, 119883, 26, 5698, 0);
INSERT INTO `history` VALUES ('2021-12-17 00:00:00', 129577, 147, 6, 3, 119916, 33, 5698, 0);
INSERT INTO `history` VALUES ('2021-12-18 00:00:00', 129678, 101, 5, 2, 119964, 48, 5698, 0);
INSERT INTO `history` VALUES ('2021-12-19 00:00:00', 129794, 116, 4, 1, 120017, 53, 5699, 1);
INSERT INTO `history` VALUES ('2021-12-20 00:00:00', 129893, 100, 1, 0, 120069, 49, 5699, 0);
INSERT INTO `history` VALUES ('2021-12-21 00:00:00', 129988, 95, 3, 2, 120146, 80, 5699, 0);
INSERT INTO `history` VALUES ('2021-12-22 00:00:00', 130139, 121, 3, 0, 120188, 42, 5699, 0);
INSERT INTO `history` VALUES ('2021-12-23 00:00:00', 130211, 112, 8, 6, 120249, 71, 5699, 0);
INSERT INTO `history` VALUES ('2021-12-24 00:00:00', 130376, 165, 10, 6, 120362, 113, 5699, 0);
INSERT INTO `history` VALUES ('2021-12-25 00:00:00', 130625, 249, 3, 0, 120438, 76, 5699, 0);
INSERT INTO `history` VALUES ('2021-12-26 00:00:00', 130858, 233, 6, 4, 120526, 53, 5699, 0);
INSERT INTO `history` VALUES ('2021-12-27 00:00:00', 131093, 235, 3, 0, 120583, 92, 5699, 0);
INSERT INTO `history` VALUES ('2021-12-28 00:00:00', 131315, 222, 3, 2, 120640, 57, 5699, 0);
INSERT INTO `history` VALUES ('2021-12-29 00:00:00', 131550, 235, 2, 2, 120701, 61, 5699, 0);
INSERT INTO `history` VALUES ('2021-12-30 00:00:00', 131780, 232, 2, 1, 120748, 47, 5699, 0);
INSERT INTO `history` VALUES ('2021-12-31 00:00:00', 132071, 291, 1, 1, 120817, 69, 5699, 0);
INSERT INTO `history` VALUES ('2022-01-01 00:00:00', 132301, 230, 1, 1, 120884, 67, 5699, 0);
INSERT INTO `history` VALUES ('2022-01-02 00:00:00', 132486, 178, 0, 0, 120936, 52, 5699, 0);
INSERT INTO `history` VALUES ('2022-01-03 00:00:00', 132692, 215, 3, 3, 120987, 51, 5699, 0);
INSERT INTO `history` VALUES ('2022-01-04 00:00:00', 132830, 148, 1, 0, 121052, 67, 5699, 0);
INSERT INTO `history` VALUES ('2022-01-05 00:00:00', 133063, 233, 2, 2, 121252, 200, 5699, 0);
INSERT INTO `history` VALUES ('2022-01-06 00:00:00', 133304, 241, 7, 6, 121354, 102, 5699, 0);
INSERT INTO `history` VALUES ('2022-01-07 00:00:00', 133540, 236, 1, 1, 121505, 151, 5699, 0);
INSERT INTO `history` VALUES ('2022-01-08 00:00:00', 133770, 230, 2, 2, 121653, 148, 5699, 0);
INSERT INTO `history` VALUES ('2022-01-09 00:00:00', 134003, 241, 3, 3, 121839, 186, 5699, 0);
INSERT INTO `history` VALUES ('2022-01-10 00:00:00', 134233, 287, 1, 0, 121989, 150, 5699, 0);
INSERT INTO `history` VALUES ('2022-01-11 00:00:00', 134540, 307, 1, 1, 122200, 211, 5699, 0);
INSERT INTO `history` VALUES ('2022-01-12 00:00:00', 134840, 298, 9, 9, 122418, 218, 5670, 1);
INSERT INTO `history` VALUES ('2022-01-13 00:00:00', 135135, 297, 13, 9, 122659, 241, 5700, 1);
INSERT INTO `history` VALUES ('2022-01-14 00:00:00', 135370, 165, 9, 6, 122816, 145, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-15 00:00:00', 135569, 199, 14, 9, 122952, 136, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-16 00:00:00', 135850, 281, 8, 4, 123169, 217, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-17 00:00:00', 136089, 239, 8, 3, 123327, 158, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-18 00:00:00', 136248, 159, 11, 5, 123520, 193, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-19 00:00:00', 136375, 127, 10, 3, 123771, 251, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-20 00:00:00', 136511, 136, 5, 1, 123989, 218, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-21 00:00:00', 136653, 142, 4, 0, 124185, 196, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-22 00:00:00', 136852, 199, 7, 3, 124409, 224, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-23 00:00:00', 137099, 247, 7, 1, 124663, 254, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-24 00:00:00', 137264, 166, 6, 0, 124876, 213, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-25 00:00:00', 137420, 156, 3, 0, 125070, 194, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-26 00:00:00', 137655, 236, 0, 0, 125290, 220, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-27 00:00:00', 137970, 315, 0, 3, 125479, 189, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-28 00:00:00', 138180, 210, 1, 0, 125702, 223, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-29 00:00:00', 138388, 208, 1, 1, 125904, 202, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-30 00:00:00', 138566, 178, 4, 3, 126074, 170, 5700, 0);
INSERT INTO `history` VALUES ('2022-01-31 00:00:00', 138753, 187, 0, 0, 126220, 146, 5700, 0);
INSERT INTO `history` VALUES ('2022-02-01 00:00:00', 138960, 207, 0, 0, 126391, 171, 5700, 0);
INSERT INTO `history` VALUES ('2022-02-02 00:00:00', 139135, 175, 0, 0, 126511, 120, 5700, 0);
INSERT INTO `history` VALUES ('2022-02-03 00:00:00', 139320, 185, 4, 4, 126662, 151, 5700, 0);
INSERT INTO `history` VALUES ('2022-02-04 00:00:00', 139640, 320, 2, 0, 126827, 165, 5700, 0);
INSERT INTO `history` VALUES ('2022-02-05 00:00:00', 140008, 369, 2, 0, 127002, 175, 5700, 0);
INSERT INTO `history` VALUES ('2022-02-06 00:00:00', 140307, 299, 1, 1, 127234, 232, 5700, 0);
INSERT INTO `history` VALUES ('2022-02-07 00:00:00', 140821, 514, 0, 0, 127468, 234, 5700, 0);
INSERT INTO `history` VALUES ('2022-02-08 00:00:00', 141128, 307, 0, 0, 127639, 171, 5700, 0);
INSERT INTO `history` VALUES ('2022-02-09 00:00:00', 141846, 718, 1, 1, 127856, 217, 5702, 2);

-- ----------------------------
-- Table structure for hotsearch
-- ----------------------------
DROP TABLE IF EXISTS `hotsearch`;
CREATE TABLE `hotsearch`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `dt` datetime NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  `content` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 575 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of hotsearch
-- ----------------------------
INSERT INTO `hotsearch` VALUES (565, '2022-02-10 16:03:15', '香港新冠确诊病例创新高');
INSERT INTO `hotsearch` VALUES (566, '2022-02-10 16:03:15', '湖南武冈市马坪乡中心小学开展疫情防控应急演练');
INSERT INTO `hotsearch` VALUES (567, '2022-02-10 16:03:15', '广西药监局从严从紧从细从实狠抓疫情防控有关工作');
INSERT INTO `hotsearch` VALUES (568, '2022-02-10 16:03:15', '北京公交集团：多个受疫情影响停运公交班次恢复运营');
INSERT INTO `hotsearch` VALUES (569, '2022-02-10 16:03:15', '黑龙江省疾控中心发布疫情防控提醒');
INSERT INTO `hotsearch` VALUES (570, '2022-02-10 16:03:15', '台湾新增83例新冠肺炎确定病例');
INSERT INTO `hotsearch` VALUES (571, '2022-02-10 16:03:15', '台湾新增37例本土新冠病例');
INSERT INTO `hotsearch` VALUES (572, '2022-02-10 16:03:15', '8日广西新增本土确诊7例');
INSERT INTO `hotsearch` VALUES (573, '2022-02-10 16:03:15', '广西德保县8216名网格员参与疫情防控');
INSERT INTO `hotsearch` VALUES (574, '2022-02-10 16:03:15', '贵州省卫健委：截至2月9日24时，现无确诊病例、无疑似病例');

-- ----------------------------
-- Table structure for sys_user
-- ----------------------------
DROP TABLE IF EXISTS `sys_user`;
CREATE TABLE `sys_user`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `password` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `email` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `phone` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of sys_user
-- ----------------------------
INSERT INTO `sys_user` VALUES (1, 'admin', '123456', '管理员', '12306@qq.com', '15655556666');

SET FOREIGN_KEY_CHECKS = 1;
