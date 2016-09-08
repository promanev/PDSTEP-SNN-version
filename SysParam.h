#define CONSTRAINT_DEBUG_SIZE 0.2f
#define TICKS_PER_DISPLAY 1

#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923
#endif

#ifndef M_PI_4
#define M_PI_4     0.785398163397448309616
#endif

// These values are obtained from:
// (1) (Sample size = 1892) "Normal Hip and Knee Active Range of Motion: The Relationship to Age", KE ROach, TP Miles, PHYS THER, 1991. Taken from Table 3, All Ages column. 
// (2) (Sample size = 537) "Normal Range of Motion of the Hip, Knee and Ankle Joints in Male Subjects, 30-40 Years of Age", A Roaas & GBJ Andersson, Acta orthop. scand. 1982. Table 1
// (3) (Sample size = 100) "Three-dimensional Lumbar Spinal Kinematics: A Study of Range of Movement in 100 Healthy Subjects Aged 20 to 60+ Years". G van Herp et al. Rheumatology, 2000. Table I, Age 20-29 averaged for F and M
#define HIP_AP_L  -(M_PI_4)*0.42222 // -19 (1)
#define HIP_AP_H  (M_PI_4)*2.68888 // 121 (1)
#define HIP_ML_L  -(M_PI_4)*0.86222 // -38.8 (2) Abduction
#define HIP_ML_H  (M_PI_4)*0.677777 // 30.5 (2) Adduction
#define ANKL_AP_L  -(M_PI_4)*0.34 // -15.3 (2) Dorsiflexion
#define ANKL_AP_H  (M_PI_4)*0.882222 // 39.7 (2) Plantar flexion
#define ANKL_ML_L  -(M_PI_4)*0.616666 // -27.75 (2) Eversion
#define ANKL_ML_H  (M_PI_4)*0.616666 // 27.75 (2) Inversion
#define TP_AP_L  -(M_PI_4)*1.281111 // -57.65 (3) Flexion
#define TP_AP_H  (M_PI_4)*0.661111 // 29.75 (3) Extension
#define TP_ML_L  -(M_PI_4)*0.565555 // -25.45 (3) Left side bending
#define TP_ML_H  (M_PI_4)*0.583333 // 26.25 (3) Left side bending
#define KNEE_AP_L  -(M_PI_4)*2.93333 // -132 (1) flexion
#define KNEE_AP_H  0 // extesion was near 0
#define KNEE_ML_L  0
#define KNEE_ML_H  0

#define INPUT_TAU  2.0
#define HIDDEN_TAU  2.0
#define OUTPUT_TAU  2.0
#define CREATE_GROUND_COLLISION_OBJECT 1
//	double avBH;
//	double avBM;
// body segments are calculated by the following formula
// y = b0 + b1 x BM + b2 x BH;

//MASSES:
//	double mass_head;
//	double mass_torso;
//	double mass_pelvis;
//	double mass_thigh;
//	double mass_shank;
//	double mass_leg; // thigh + shank
//	double mass_foot;
//	double mass_UA; // upper arm separately
//	double mass_FA;  // forearm separately
//	double mass_arm; // UA+FA
//	double mass_hand;

	// HEIGHTS:
//	double height_head;
//	double height_torso; // upper + middle torso
//	double height_pelvis;
//	double height_thigh;
//	double height_shank;
//	double height_leg; // thigh + shank
//	double length_foot;
//	double height_UA; // upper arm separately
//	double height_FA;  // forearm separately
//	double height_arm; // UA+FA
//	double height_hand;

//#ifdef MALE
//  avBH = 181.0;
//	avBM = 78.4;
// body segments are calculated by the following formula
// y = b0 + b1 x BM + b2 x BH;

//MASSES:
//	mass_head = -7.75 + 0.0586*avBM + 0.0497*avBH;
//	mass_torso = 7.57 + 0.295*avBM - 0.0385*avBH; // upper + middle torso
//	mass_pelvis = 13.1 + 0.162*avBM - 0.0873*avBH;
//	mass_thigh = 1.18 + 0.182*avBM - 0.0259*avBH;
//	mass_shank = -3.53 + 0.0306*avBM + 0.0268*avBH;
//	mass_leg = -2.35 + 0.2126*avBM + 0.0009*avBH; // thigh + shank
//	mass_foot = -2.25 + 0.0010*avBM + 0.0182*avBH;
//	mass_UA = -0.896 + 0.0252*avBM + 0.0051*avBH; // upper arm separately
//	mass_FA = -0.731 + 0.0047*avBM + 0.0084*avBH;  // forearm separately
//	mass_arm = -1.627 + 0.0299*avBM + 0.0135*avBH; // UA+FA
//	mass_hand = -0.325 - 0.0016*avBM + 0.0051*avBH;

	// HEIGHTS:
//	height_head = 1.95 + 0.0535*avBM + 0.105*avBH;
//	height_torso = -32.11 - 0.095*avBM + 0.462*avBH; // upper + middle torso
//	height_pelvis = 26.4 + 0.0473*avBM - 0.0311*avBH;
//	height_thigh = 4.26 - 0.0183*avBM + 0.24*avBH;
//	height_shank = -16.0 + 0.0218*avBM + 0.321*avBH;
//	height_leg = -11.74 + 0.0035*avBM + 0.561*avBH; // thigh + shank
//	length_foot = 3.8 + 0.013*avBM + 0.119*avBH;
//	height_UA = -15.0 + 0.012*avBM + 0.229*avBH; // upper arm separately
//	height_FA = 0.143 - 0.0281*avBM + 0.161*avBH;  // forearm separately
//	height_arm = -14.857 - 0.0161*avBM + 0.39*avBH; // UA+FA
//	height_hand = -3.7 + 0.0036*avBM + 0.131*avBH;

//#else //Female:
//	avBH = 169.0;
//	avBM = 75.4;

//MASSES:
//	mass_head = -2.95 + 0.0359*avBM + 0.0322*avBH;
//	mass_torso = 24.05 + 0.3255*avBM - 0.1424*avBH; // upper + middle torso
//	mass_pelvis = 1.1 + 0.104*avBM - 0.0027*avBH;
//	mass_thigh = -10.9 + 0.213*avBM + 0.038*avBH;
//	mass_shank = -0.563 + 0.0191*avBM + 0.0141*avBH;
//	mass_leg = mass_thigh + mass_shank; // thigh + shank
//	mass_foot = -1.27 + 0.0045*avBM + 0.0104*avBH;
//	mass_UA = 3.05 + 0.0184*avBM - 0.0164*avBH; // upper arm separately
//	mass_FA = -0.481 + 0.0087*avBM + 0.0043*avBH;  // forearm separately
//	mass_arm = mass_UA + mass_FA; // UA+FA
//	mass_hand = -1.13 + 0.0031*avBM + 0.0074*avBH;

// HEIGHTS:
//	height_head = -8.95 - 0.0057*avBM + 0.202*avBH;
//	height_torso = 10.48 + 0.1291*avBM + 0.147*avBH; // upper + middle torso
//	height_pelvis = 21.4 + 0.0146*avBM - 0.005*avBH;
//	height_thigh = -26.8 - 0.0725*avBM + 0.436*avBH;
//	height_shank = -7.21 - 0.0618*avBM + 0.308*avBH;
//	height_leg = height_thigh + height_shank; // thigh + shank
//	length_foot = 7.39 + 0.0311*avBM + 0.0867*avBH;
//	height_UA = 2.44 - 0.0169*avBM + 0.146*avBH; // upper arm separately
//	height_FA = -8.57 + 0.0494*avBM + 0.18*avBH;  // forearm separately
//	height_arm = height_FA + height_UA; // UA+FA
//	height_hand = -8.96 + 0.0057*avBM + 0.163*avBH;
//#endif


