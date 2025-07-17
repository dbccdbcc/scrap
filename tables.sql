-- Table racedates
CREATE TABLE `racedates` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `RaceDate` DATE DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Table race_results
CREATE TABLE `race_results` (
  `id` int NOT NULL AUTO_INCREMENT,
  `race_date` date DEFAULT NULL,
  `course` varchar(10) DEFAULT NULL,
  `race_no` int DEFAULT NULL,
  `race_info` varchar(255) DEFAULT NULL,
  `pla` varchar(10) DEFAULT NULL,
  `horse_no` varchar(10) DEFAULT NULL,
  `horse` varchar(100) DEFAULT NULL,
  `jockey` varchar(100) DEFAULT NULL,
  `trainer` varchar(100) DEFAULT NULL,
  `act_wt` varchar(10) DEFAULT NULL,
  `declared_wt` varchar(10) DEFAULT NULL,
  `draw_no` varchar(10) DEFAULT NULL,
  `lbw` varchar(10) DEFAULT NULL,
  `running_position` varchar(50) DEFAULT NULL,
  `finish_time` varchar(20) DEFAULT NULL,
  `win_odds` varchar(10) DEFAULT NULL,
  `url` text,
  `raceDateId` int DEFAULT NULL,
  `race_class` varchar(100) DEFAULT NULL,
  `distance` int DEFAULT NULL,
  `surface` varchar(100) DEFAULT NULL,
  `track` varchar(100) DEFAULT NULL,
  `going` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_race_results_raceDateId` (`raceDateId`),
  CONSTRAINT `fk_race_results_raceDateId` FOREIGN KEY (`raceDateId`) REFERENCES `racedates` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=170242 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Table future_races
CREATE TABLE `future_races` (
  `id` int NOT NULL AUTO_INCREMENT,
  `race_info` varchar(100) DEFAULT NULL,
  `horse_no` int DEFAULT NULL,
  `horse` varchar(100) DEFAULT NULL,
  `draw_no` int DEFAULT NULL,
  `act_wt` int DEFAULT NULL,
  `jockey` varchar(100) DEFAULT NULL,
  `trainer` varchar(100) DEFAULT NULL,
  `win_odds` float DEFAULT NULL,
  `place_odds` float DEFAULT NULL,
  `race_date` date DEFAULT NULL,
  `course` varchar(50) DEFAULT NULL,
  `race_no` int DEFAULT NULL,
  `race_class` varchar(100) DEFAULT NULL,
  `distance` int DEFAULT NULL,
  `surface` varchar(100) DEFAULT NULL,
  `track` varchar(100) DEFAULT NULL,
  `going` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=109 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Table race_predictions (model + simulation output)
CREATE TABLE `race_predictions` (
  `race_date` DATETIME DEFAULT NULL,
  `race_no` BIGINT DEFAULT NULL,
  `horse_no` BIGINT DEFAULT NULL,
  `horse` TEXT,
  `win_probability` FLOAT DEFAULT NULL,
  `place_probability` FLOAT DEFAULT NULL,
  `sim_win_pct` FLOAT DEFAULT NULL,
  `sim_place_pct` FLOAT DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
