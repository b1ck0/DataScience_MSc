\* Juice Blending Problem *\
Minimize
total_cost_of_oranges: 675 Suppliers_Arizona + 600 Suppliers_Brazil
 + 660 Suppliers_California + 560 Suppliers_China + 590 Suppliers_Egypt
 + 750 Suppliers_Florida + 460 Suppliers_India + 620 Suppliers_Italy
 + 390 Suppliers_Mexico + 710 Suppliers_Spain + 675 Suppliers_Texas
Subject To
Lower_Acid_constraint: - 0.0005 Suppliers_Arizona - 0.0015 Suppliers_Brazil
 + 0.0025 Suppliers_California + 0.006 Suppliers_China
 + 0.0055 Suppliers_Egypt + 0.002 Suppliers_Florida + 0.0065 Suppliers_India
 - 0.0025 Suppliers_Italy + 0.0055 Suppliers_Mexico + 0.0035 Suppliers_Spain
 - 0.0005 Suppliers_Texas >= 0
Lower_Astringency_constraint: Suppliers_Arizona + 3 Suppliers_Brazil
 + 3 Suppliers_California + 7 Suppliers_China + 3 Suppliers_Egypt
 + 3 Suppliers_Florida + 7 Suppliers_India + 3 Suppliers_Italy
 + 8 Suppliers_Mexico + 4 Suppliers_Spain + Suppliers_Texas >= 0
Lower_BAR_constraint: 0.5 Suppliers_Arizona - Suppliers_Brazil
 - 0.5 Suppliers_California - 2.5 Suppliers_China + 1.5 Suppliers_Egypt
 + 0.5 Suppliers_Florida - 5 Suppliers_India + 2.5 Suppliers_Italy
 - 3.5 Suppliers_Mexico + 3.5 Suppliers_Spain - 1.5 Suppliers_Texas >= 0
Lower_Color_constraint: 0.5 Suppliers_Arizona - 1.5 Suppliers_Brazil
 + 0.5 Suppliers_California - 1.5 Suppliers_China + 0.5 Suppliers_Egypt
 - 1.5 Suppliers_Florida - 3.5 Suppliers_India + 4.5 Suppliers_Italy
 - 1.5 Suppliers_Mexico + 3.5 Suppliers_Spain + 0.5 Suppliers_Texas >= 0
Upper_Acid_constraint: - 0.003 Suppliers_Arizona - 0.004 Suppliers_Brazil
 + 0 Suppliers_California + 0.0035 Suppliers_China + 0.003 Suppliers_Egypt
 - 0.0005 Suppliers_Florida + 0.004 Suppliers_India - 0.005 Suppliers_Italy
 + 0.003 Suppliers_Mexico + 0.001 Suppliers_Spain - 0.003 Suppliers_Texas <= 0
Upper_Astringency_constraint: - 3 Suppliers_Arizona - Suppliers_Brazil
 - Suppliers_California + 3 Suppliers_China - Suppliers_Egypt
 - Suppliers_Florida + 3 Suppliers_India - Suppliers_Italy
 + 4 Suppliers_Mexico + 0 Suppliers_Spain - 3 Suppliers_Texas <= 0
Upper_BAR_constraint: - 0.5 Suppliers_Arizona - 2 Suppliers_Brazil
 - 1.5 Suppliers_California - 3.5 Suppliers_China + 0.5 Suppliers_Egypt
 - 0.5 Suppliers_Florida - 6 Suppliers_India + 1.5 Suppliers_Italy
 - 4.5 Suppliers_Mexico + 2.5 Suppliers_Spain - 2.5 Suppliers_Texas <= 0
Upper_Color_constraint: - 0.5 Suppliers_Arizona - 2.5 Suppliers_Brazil
 - 0.5 Suppliers_California - 2.5 Suppliers_China - 0.5 Suppliers_Egypt
 - 2.5 Suppliers_Florida - 4.5 Suppliers_India + 3.5 Suppliers_Italy
 - 2.5 Suppliers_Mexico + 2.5 Suppliers_Spain - 0.5 Suppliers_Texas <= 0
