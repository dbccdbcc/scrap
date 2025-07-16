-- Table racedates
CREATE TABLE `racedates` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `RaceDate` DATE DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Table race_results
CREATE TABLE `race_results` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `race_date` DATE DEFAULT NULL,
  `course` VARCHAR(10) DEFAULT NULL,
  `race_no` INT DEFAULT NULL,
  `race_info` VARCHAR(255) DEFAULT NULL,
  `pla` VARCHAR(10) DEFAULT NULL,
  `horse_no` VARCHAR(10) DEFAULT NULL,
  `horse` VARCHAR(100) DEFAULT NULL,
  `jockey` VARCHAR(100) DEFAULT NULL,
  `trainer` VARCHAR(100) DEFAULT NULL,
  `act_wt` VARCHAR(10) DEFAULT NULL,
  `declared_wt` VARCHAR(10) DEFAULT NULL,
  `draw_no` VARCHAR(10) DEFAULT NULL,
  `lbw` VARCHAR(10) DEFAULT NULL,
  `running_position` VARCHAR(50) DEFAULT NULL,
  `finish_time` VARCHAR(20) DEFAULT NULL,
  `win_odds` VARCHAR(10) DEFAULT NULL,
  `url` TEXT,
  `raceDateId` INT DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_race_results_raceDateId` (`raceDateId`),
  CONSTRAINT `fk_race_results_raceDateId` FOREIGN KEY (`raceDateId`) REFERENCES `racedates` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Table future_races
CREATE TABLE `future_races` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `race_info` VARCHAR(100) DEFAULT NULL,
  `horse_no` INT DEFAULT NULL,
  `horse` VARCHAR(100) DEFAULT NULL,
  `draw_no` INT DEFAULT NULL,
  `act_wt` INT DEFAULT NULL,
  `jockey` VARCHAR(100) DEFAULT NULL,
  `trainer` VARCHAR(100) DEFAULT NULL,
  `win_odds` FLOAT DEFAULT NULL,
  `place_odds` FLOAT DEFAULT NULL,
  `race_date` DATE DEFAULT NULL,
  `course` VARCHAR(50) DEFAULT NULL,
  `race_no` INT DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

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
