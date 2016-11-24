/*
Modified from original Ragdoll Demo by Roman Popov with help from Anton Bernadsky.
June 2013 - September 2016
University of Vermont, Burlington, VT, USA.
rpopov@uvm.edu, promanev@gmail.com
*/

#include "btBulletDynamicsCommon.h"
#include "GlutStuff.h"
#include "GL_ShapeDrawer.h"
#include "GlutDemoApplication.h"
// added for CTRNN
#include <iomanip>

// old added
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <time.h>
#include <math.h>
#include <random>
#include <tuple>
using namespace std;
//end added

#include "LinearMath/btIDebugDraw.h"

#include "GLDebugDrawer.h"
#include "PDSTEP_demo.h"
#include "SysParam.h"


class RagDoll
{
#ifdef MALE
	double avBH = 181.0;
	double avBM = 78.4;
	// body segments are calculated by the following formula
	// y = b0 + b1 x BM + b2 x BH;

	//MASSES:
	double mass_head = -7.75 + 0.0586*avBM + 0.0497*avBH;
	double mass_torso = 7.57 + 0.295*avBM - 0.0385*avBH; // upper + middle torso
	double mass_pelvis = 13.1 + 0.162*avBM - 0.0873*avBH;
	double mass_thigh = 1.18 + 0.182*avBM - 0.0259*avBH;
	double mass_shank = -3.53 + 0.0306*avBM + 0.0268*avBH;
	double mass_leg = -2.35 + 0.2126*avBM + 0.0009*avBH; // thigh + shank
	double mass_foot = -2.25 + 0.0010*avBM + 0.0182*avBH;
	double mass_UA = -0.896 + 0.0252*avBM + 0.0051*avBH; // upper arm separately
	double mass_FA = -0.731 + 0.0047*avBM + 0.0084*avBH;  // forearm separately
	double mass_arm = -1.627 + 0.0299*avBM + 0.0135*avBH; // UA+FA
	double mass_hand = -0.325 - 0.0016*avBM + 0.0051*avBH; 

//	// HEIGHTS:
	double height_head = 1.95 + 0.0535*avBM + 0.105*avBH;
	double height_torso = -32.11 - 0.095*avBM + 0.462*avBH; // upper + middle torso
	double height_pelvis = 26.4 + 0.0473*avBM - 0.0311*avBH;
	double height_thigh = 4.26 - 0.0183*avBM + 0.24*avBH;
	double height_shank = -16.0 + 0.0218*avBM + 0.321*avBH;
	double height_leg = -11.74 + 0.0035*avBM + 0.561*avBH; // thigh + shank
	double length_foot = 3.8 + 0.013*avBM + 0.119*avBH;
	double height_UA = -15.0 + 0.012*avBM + 0.229*avBH; // upper arm separately
	double height_FA = 0.143 - 0.0281*avBM + 0.161*avBH;  // forearm separately
	double height_arm = -14.857 - 0.0161*avBM + 0.39*avBH; // UA+FA
	double height_hand = -3.7 + 0.0036*avBM + 0.131*avBH;//

#else //Female:
	double avBH = 169.0;
	double avBM = 75.4;

//	//MASSES:
	double mass_head = -2.95 + 0.0359*avBM + 0.0322*avBH;
	double mass_torso = 24.05 + 0.3255*avBM - 0.1424*avBH; // upper + middle torso
	double mass_pelvis = 1.1 + 0.104*avBM - 0.0027*avBH;
	double mass_thigh = -10.9 + 0.213*avBM + 0.038*avBH;
	double mass_shank = -0.563 + 0.0191*avBM + 0.0141*avBH;
	double mass_leg = mass_thigh + mass_shank; // thigh + shank
	double mass_foot = -1.27 + 0.0045*avBM + 0.0104*avBH;
	double mass_UA = 3.05 + 0.0184*avBM - 0.0164*avBH; // upper arm separately
	double mass_FA = -0.481 + 0.0087*avBM + 0.0043*avBH;  // forearm separately
	double mass_arm = mass_UA + mass_FA; // UA+FA
	double mass_hand = -1.13 + 0.0031*avBM + 0.0074*avBH;//

//	// HEIGHTS:
	double height_head = -8.95 - 0.0057*avBM + 0.202*avBH;
	double height_torso = 10.48 + 0.1291*avBM + 0.147*avBH; // upper + middle torso
	double height_pelvis = 21.4 + 0.0146*avBM - 0.005*avBH;
	double height_thigh = -26.8 - 0.0725*avBM + 0.436*avBH;
	double height_shank = -7.21 - 0.0618*avBM + 0.308*avBH;
	double height_leg = height_thigh + height_shank; // thigh + shank
	double length_foot = 7.39 + 0.0311*avBM + 0.0867*avBH;
	double height_UA = 2.44 - 0.0169*avBM + 0.146*avBH; // upper arm separately
	double height_FA = -8.57 + 0.0494*avBM + 0.18*avBH;  // forearm separately
	double height_arm = height_FA + height_UA; // UA+FA
	double height_hand = -8.96 + 0.0057*avBM + 0.163*avBH;
#endif

	// for the case of torso
#ifdef TORSO
		enum
		{
			BODYPART_ABDOMEN = 0, //1
			BODYPART_PELVIS, //2
			BODYPART_LEFT_LEG, //3
			BODYPART_RIGHT_LEG, //4
			BODYPART_LEFT_FOOT, //5
			BODYPART_RIGHT_FOOT, //6
			BODYPART_PLATFORM, //7

			BODYPART_COUNT
		};
#else
#ifndef KNEES
	enum
	{
		
		BODYPART_PELVIS = 0, //1 
		BODYPART_LEFT_LEG, //2
		BODYPART_RIGHT_LEG, //3
		BODYPART_LEFT_FOOT, //4
		BODYPART_RIGHT_FOOT, //5
		BODYPART_PLATFORM, //6

		BODYPART_COUNT
	};
#else
	enum
	{
		BODYPART_PELVIS = 0, //1
		BODYPART_LEFT_THIGH, //2
		BODYPART_RIGHT_THIGH, //3
		BODYPART_LEFT_SHANK, //4
		BODYPART_RIGHT_SHANK, //5
		BODYPART_LEFT_FOOT, //6
		BODYPART_RIGHT_FOOT, //7
		BODYPART_PLATFORM, //8

		BODYPART_COUNT
	};
#endif
#endif

	// in the case of torso - one more joint
#ifdef TORSO
		enum
		{
			JOINT_LEFT_HIP=0, //1
			JOINT_RIGHT_HIP, //2
			JOINT_LEFT_ANKLE, //3
			JOINT_RIGHT_ANKLE, //4
			JOINT_BODY_PELVIS, //5

			JOINT_COUNT
		};

#else
#ifndef KNEES 
	enum
	{
		
		JOINT_LEFT_HIP = 0, //1	
		JOINT_RIGHT_HIP, //2
		JOINT_LEFT_ANKLE, //3
		JOINT_RIGHT_ANKLE, //4

		JOINT_COUNT
	};
#else
	enum
	{
		JOINT_LEFT_HIP = 0, //1
		JOINT_RIGHT_HIP, //2
		JOINT_LEFT_ANKLE, //3
		JOINT_RIGHT_ANKLE, //4
		JOINT_LEFT_KNEE, //5
		JOINT_RIGHT_KNEE, //6

		JOINT_COUNT
	};
#endif
#endif
	enum legOrient
	{
		X_ORIENT,
		Y_ORIENT,
		Z_ORIENT
	} orient;

	enum jointOrient
	{
		X_ORIENT_P, //positive orientations
		Y_ORIENT_P,
		Z_ORIENT_P,
		X_ORIENT_N, //negative orientations
		Y_ORIENT_N,
		Z_ORIENT_N
	} j_orient;

	// USED TO BE PRIVATE:
	btDynamicsWorld* m_ownerWorld;
	btCollisionShape* m_shapes[BODYPART_COUNT];
	btRigidBody* m_bodies[BODYPART_COUNT];
	// btTypedConstraint* m_joints[JOINT_COUNT]; // replaced by line below
	btGeneric6DofConstraint* m_joints[JOINT_COUNT];
	btJointFeedback fg;
	// END USED TO BE PRIVATE

	int * m_ids;

	btRigidBody* localCreateRigidBody (btScalar mass, const btTransform& startTransform, btCollisionShape* shape)
	{
		bool isDynamic = (mass != 0.f);

		btVector3 localInertia(0,0,0);
		if (isDynamic)
			shape->calculateLocalInertia(mass,localInertia);

		btDefaultMotionState* myMotionState = new btDefaultMotionState(startTransform);
		
		btRigidBody::btRigidBodyConstructionInfo rbInfo(mass,myMotionState,shape,localInertia);
		btRigidBody* body = new btRigidBody(rbInfo);

		m_ownerWorld->addRigidBody(body);

		return body;
	}

public:
	//new feet vals:
	double footLen = length_foot / 60;
#ifdef OLD_FEET //preserve old code
	double footWid = length_foot / 90;
	double footHeight = length_foot / 120;
#else
	double footWid = footLen * 0.375;
	double footHeight = footLen * 0.375;
#endif

	RagDoll (btDynamicsWorld* ownerWorld, const btVector3& positionOffset, int* IDs)
		: m_ownerWorld(ownerWorld), m_ids(IDs)
	{
		//CREATE BOXES:
		//First three numbers-COM location, second 3 numbers - half-measures (since the origin is inside the body's center)
		//of the body itself

		//in case of the torso
#ifdef TORSO
		CreateBox(BODYPART_PLATFORM, 0, 0.15, 0, 6., 4., 0.15, 200.);
		// mixed up dimensions: width, height, length, now corrected to that described: length, width, height
		// width - mediolateral dir, length into the screen

		//all heights are scaled down by 30, to be comparable with the previous robot. Also units can be thought of as feet
		// since 1 foot = 30.4878 cm. Values are parsed divided by 60 because functions take in half-measures. 
		CreateBox(BODYPART_ABDOMEN, 0., 0.3+length_foot/60+height_leg/30+height_pelvis/60+height_torso/60, 0., height_pelvis/60, height_pelvis/60, height_torso/60, mass_torso);
		CreateBox(BODYPART_PELVIS, 0., 0.3+length_foot/60+height_leg/30, 0., height_pelvis/60, height_pelvis/60, height_pelvis/60, mass_pelvis);
		CreateBox(BODYPART_LEFT_FOOT, height_pelvis/60, 0.3+length_foot/120, 0., length_foot/ 60, length_foot/90, length_foot/120, mass_foot);
		CreateBox(BODYPART_RIGHT_FOOT, -height_pelvis/60, 0.3+length_foot/120, 0., length_foot/60, length_foot/90, length_foot/120, mass_foot);
#else	
#ifndef KNEES
		CreateBox(BODYPART_PLATFORM, 0, 0.15, 0, 6., 4., 0.15, 200.);
		CreateBox(BODYPART_PELVIS, 0., 0.3+length_foot/60+height_leg/30, 0., height_pelvis/60, height_pelvis/60, height_pelvis/60, mass_pelvis);
		CreateBox(BODYPART_LEFT_FOOT, height_pelvis/60, 0.3+length_foot/120, 0., length_foot/60, length_foot/90, length_foot/120, mass_foot);
		CreateBox(BODYPART_RIGHT_FOOT, -height_pelvis/60, 0.3+length_foot/120, 0., length_foot/60, length_foot/90, length_foot/120, mass_foot);
#else
		CreateBox(BODYPART_PLATFORM, 0, 0.15, 0, 6., 4., 0.15, 200.);
		CreateBox(BODYPART_PELVIS, 0., 0.3 + footHeight*2 + height_leg / 30, 0., height_pelvis / 60, height_pelvis / 60, height_pelvis / 60, mass_pelvis);
		//CreateBox(BODYPART_LEFT_FOOT, height_pelvis / 60, 0.3 + length_foot / 120, 0., length_foot / 60, length_foot / 90, length_foot / 120, mass_foot);
		//CreateBox(BODYPART_RIGHT_FOOT, -height_pelvis / 60, 0.3 + length_foot / 120, 0., length_foot / 60, length_foot / 90, length_foot / 120, mass_foot);
		
		//new feet proportions (based on my foot): length = 1; width, height = 0.375;
		CreateBox(BODYPART_LEFT_FOOT, height_pelvis / 60, 0.3 + footHeight, 0., footLen, footWid, footHeight, mass_foot);
		CreateBox(BODYPART_RIGHT_FOOT, -height_pelvis / 60, 0.3 + footHeight, 0., footLen, footWid, footHeight, mass_foot);
#endif
#endif	


		//CREATE LEGS:
#ifndef KNEES
		CreateCylinder(BODYPART_LEFT_LEG, Y_ORIENT, height_pelvis/60, 0.3+length_foot/60+height_leg/60, 0., height_leg/60, 0.15, 1., mass_leg);
		CreateCylinder(BODYPART_RIGHT_LEG, Y_ORIENT, -height_pelvis/60, 0.3+length_foot/60+height_leg/60, 0., height_leg/60, 0.15, 1., mass_leg);
#else
		CreateCylinder(BODYPART_LEFT_THIGH, Y_ORIENT, height_pelvis / 60, 0.3 + footHeight * 2 + height_shank / 30 + height_thigh/60, 0., height_thigh / 60, 0.15, 1., mass_thigh);
		CreateCylinder(BODYPART_RIGHT_THIGH, Y_ORIENT, -height_pelvis / 60, 0.3 + footHeight * 2 + height_shank / 30 + height_thigh / 60, 0., height_thigh / 60, 0.15, 1., mass_thigh);
		CreateCylinder(BODYPART_LEFT_SHANK, Y_ORIENT, height_pelvis / 60, 0.3 + footHeight * 2 + height_shank / 60, 0., height_shank / 60, 0.15, 1., mass_shank);
		CreateCylinder(BODYPART_RIGHT_SHANK, Y_ORIENT, -height_pelvis / 60, 0.3 + footHeight * 2 + height_shank / 60, 0., height_shank / 60, 0.15, 1., mass_shank);
#endif
		//CREATE JOINTS:
		//vectors in argument are the joint location in local body part's coordinate system

#ifdef TORSO

		//Flipped the boundary values on 25.02.2014, used to be 2.5 and 0.5 for AL, ML is the same - 0.67 and 0.67. Flipped again on 3.03.2014, researching the bug where the limits on the targetAngle produced a "jumping" bug. 
		Create6DOF(JOINT_LEFT_HIP, BODYPART_PELVIS, BODYPART_LEFT_LEG, btVector3(height_pelvis/60, 0., 0.), btVector3(0., height_leg/60, 0.), M_PI_2, 0, M_PI_2, HIP_AP_L,HIP_AP_H,HIP_ML_L,HIP_ML_H);
		Create6DOF(JOINT_RIGHT_HIP, BODYPART_PELVIS, BODYPART_RIGHT_LEG, btVector3(-height_pelvis/60, 0., 0.), btVector3(0., height_leg/60, 0.), M_PI_2, 0, M_PI_2, HIP_AP_L, HIP_AP_H, HIP_ML_L, HIP_ML_H);
		//Flipped the boundary values on 25.02.2014, used to be 1.3 and 0.3 for AL. Flipped again on 3.03.2014, researching the bug where the limits on the targetAngle produced a "jumping" bug.  
		Create6DOF(JOINT_LEFT_ANKLE, BODYPART_LEFT_LEG, BODYPART_LEFT_FOOT, btVector3(0.,-height_leg/60, 0.), btVector3(0.,length_foot/120, 0.), M_PI_2, 0, M_PI_2, ANKL_AP_L, ANKL_AP_H, ANKL_ML_L, ANKL_ML_H);
		//28 degrees (0.62*(pi/4=45degrees)) for ML movement in the ankles 
		Create6DOF(JOINT_RIGHT_ANKLE, BODYPART_RIGHT_LEG, BODYPART_RIGHT_FOOT, btVector3(0.,-height_leg/60, 0.), btVector3(0.,length_foot/120, 0.), M_PI_2, 0, M_PI_2, ANKL_AP_L, ANKL_AP_H, ANKL_ML_L, ANKL_ML_H);
		Create6DOF(JOINT_BODY_PELVIS, BODYPART_ABDOMEN, BODYPART_PELVIS, btVector3(0., -height_torso/60, height_pelvis/120), btVector3(0., height_pelvis/60, height_pelvis/120), M_PI_2, 0, M_PI_2, TP_AP_L, TP_AP_H, TP_ML_L, TP_ML_H);
#else
#ifndef KNEES
		//Flipped the boundary values on 25.02.2014, used to be 2.5 and 0.5 for AL, ML is the same - 0.67 and 0.67. Flipped again on 3.03.2014, researching the bug where the limits on the targetAngle produced a "jumping" bug. 
		Create6DOF(JOINT_LEFT_HIP, BODYPART_PELVIS, BODYPART_LEFT_LEG, btVector3(height_pelvis/60, 0., 0.), btVector3(0., height_leg/60, 0.), M_PI_2, 0, M_PI_2, HIP_AP_L, HIP_AP_H, HIP_ML_L, HIP_ML_H);
		Create6DOF(JOINT_RIGHT_HIP, BODYPART_PELVIS, BODYPART_RIGHT_LEG, btVector3(-height_pelvis/60, 0., 0.), btVector3(0., height_leg/60, 0.), M_PI_2, 0, M_PI_2, HIP_AP_L, HIP_AP_H, HIP_ML_L, HIP_ML_H);
		//Flipped the boundary values on 25.02.2014, used to be 1.3 and 0.3 for AL. Flipped again on 3.03.2014, researching the bug where the limits on the targetAngle produced a "jumping" bug.  
		Create6DOF(JOINT_LEFT_ANKLE, BODYPART_LEFT_LEG, BODYPART_LEFT_FOOT, btVector3(0., -height_leg/60, 0.), btVector3(0., length_foot/120, 0.), M_PI_2, 0, M_PI_2, ANKL_AP_L, ANKL_AP_H, ANKL_ML_L, ANKL_ML_H);
		//28 degrees (0.62*(pi/4=45degrees)) for ML movement in the ankles 
		Create6DOF(JOINT_RIGHT_ANKLE, BODYPART_RIGHT_LEG, BODYPART_RIGHT_FOOT, btVector3(0., -height_leg/60, 0.), btVector3(0., length_foot/120, 0.), M_PI_2, 0, M_PI_2, ANKL_AP_L, ANKL_AP_H, ANKL_ML_L, ANKL_ML_H);
#else // if KNEES:
		Create6DOF(JOINT_LEFT_HIP, BODYPART_PELVIS, BODYPART_LEFT_THIGH, btVector3(height_pelvis / 60, 0., 0.), btVector3(0., height_thigh / 60, 0.), M_PI_2, 0, M_PI_2, HIP_AP_L, HIP_AP_H, HIP_ML_L, HIP_ML_H);
		Create6DOF(JOINT_RIGHT_HIP, BODYPART_PELVIS, BODYPART_RIGHT_THIGH, btVector3(-height_pelvis / 60, 0., 0.), btVector3(0., height_thigh / 60, 0.), M_PI_2, 0, M_PI_2, HIP_AP_L, HIP_AP_H, HIP_ML_L, HIP_ML_H);
		Create6DOF(JOINT_LEFT_KNEE, BODYPART_LEFT_THIGH, BODYPART_LEFT_SHANK, btVector3(0., -height_thigh / 60, 0.), btVector3(0., height_shank / 60, 0.), M_PI_2, 0, M_PI_2, KNEE_AP_L, KNEE_AP_H, KNEE_ML_L, KNEE_ML_H);
		Create6DOF(JOINT_RIGHT_KNEE, BODYPART_RIGHT_THIGH, BODYPART_RIGHT_SHANK, btVector3(0., -height_thigh / 60, 0.), btVector3(0., height_shank / 60, 0.), M_PI_2, 0, M_PI_2, KNEE_AP_L, KNEE_AP_H, KNEE_ML_L, KNEE_ML_H);
		// new position of ankle joints:
		Create6DOF(JOINT_LEFT_ANKLE, BODYPART_LEFT_SHANK, BODYPART_LEFT_FOOT, btVector3(0., -height_shank / 60, 0.), btVector3(0., footHeight, 0.), M_PI_2, 0, M_PI_2, ANKL_AP_L, ANKL_AP_H, ANKL_ML_L, ANKL_ML_H);
		Create6DOF(JOINT_RIGHT_ANKLE, BODYPART_RIGHT_SHANK, BODYPART_RIGHT_FOOT, btVector3(0., -height_shank / 60, 0.), btVector3(0., footHeight, 0.), M_PI_2, 0, M_PI_2, ANKL_AP_L, ANKL_AP_H, ANKL_ML_L, ANKL_ML_H);
		
#endif
#endif
		//FRICTION CTRL:
		m_bodies[BODYPART_LEFT_FOOT]->setFriction(7); 
		m_bodies[BODYPART_RIGHT_FOOT]->setFriction(7);
		m_bodies[BODYPART_PLATFORM]->setFriction(7);
		return;
	}

	// DESTRUCTOR:
	virtual	~RagDoll ()
	{
		int i;

		// Remove all constraints
		for ( i = 0; i < JOINT_COUNT; ++i)
		{
			m_ownerWorld->removeConstraint(m_joints[i]);
			delete m_joints[i]; m_joints[i] = 0;
		}

		// Remove all bodies and shapes
		for ( i = 0; i < BODYPART_COUNT; ++i)
		{
			m_ownerWorld->removeRigidBody(m_bodies[i]);			
			delete m_bodies[i]->getMotionState();
			delete m_bodies[i]; m_bodies[i] = 0;
			delete m_shapes[i]; m_shapes[i] = 0;
		}
	}
	// CREATE BOX:
	void CreateBox(int index, double x, double y, double z, double length, double width, double height, double mass)
	{
		btVector3 positionOffset(0., 0., 0.);
		//m_shapes[index] = new btBoxShape(btVector3(length, width, height));
		m_shapes[index] = new btBoxShape(btVector3(width, height, length));
		btTransform offset; offset.setIdentity();
		offset.setOrigin(positionOffset);

		btTransform transform;

		transform.setIdentity();
		transform.setOrigin(btVector3(btScalar(x), btScalar(y), btScalar(z)));
		m_bodies[index] = localCreateRigidBody(btScalar(mass), offset*transform, m_shapes[index]);
		m_bodies[index]->setDamping(0.05, 0.85);
		//m_bodies[index]->setDeactivationTime(0.8);
		//m_bodies[index]->setSleepingThresholds(1.6, 2.5);
		(m_bodies[index])->setUserPointer(&(m_ids[index]));
		m_bodies[index]->setActivationState(DISABLE_DEACTIVATION);

	}
	// CREATE CYLINDER:
	void CreateCylinder(int index, legOrient orient, double x, double y, double z, double length, double width, double height, double mass)
	{
		switch (orient)
		{
		case X_ORIENT:
		{btVector3 positionOffset(0., 0., 0.);
		m_shapes[index] = new btCylinderShapeX(btVector3(width, length, height));
		btTransform offset; offset.setIdentity();
		offset.setOrigin(positionOffset);

		btTransform transform;

		transform.setIdentity();
		transform.setOrigin(btVector3(btScalar(x), btScalar(y), btScalar(z)));
		m_bodies[index] = localCreateRigidBody(btScalar(mass), offset*transform, m_shapes[index]);
		m_bodies[index]->setActivationState(DISABLE_DEACTIVATION);
		m_bodies[index]->setDamping(0.05, 0.85);
		//m_bodies[index]->setDeactivationTime(0.8);
		//m_bodies[index]->setSleepingThresholds(1.6, 2.5);
		break;
		}
		case Y_ORIENT:
		{btVector3 positionOffset(0., 0., 0.);
		m_shapes[index] = new btCylinderShape(btVector3(width, length, height));
		btTransform offset; offset.setIdentity();
		offset.setOrigin(positionOffset);

		btTransform transform;

		transform.setIdentity();
		transform.setOrigin(btVector3(btScalar(x), btScalar(y), btScalar(z)));
		m_bodies[index] = localCreateRigidBody(btScalar(mass), offset*transform, m_shapes[index]);
		m_bodies[index]->setDamping(0.05, 0.85);
		//m_bodies[index]->setDeactivationTime(0.8);
		//m_bodies[index]->setSleepingThresholds(1.6, 2.5);
		break;
		}
		case Z_ORIENT:
		{btVector3 positionOffset(0., 0., 0.);
		m_shapes[index] = new btCylinderShapeZ(btVector3(width, length, height));
		btTransform offset; offset.setIdentity();
		offset.setOrigin(positionOffset);

		btTransform transform;

		transform.setIdentity();
		transform.setOrigin(btVector3(btScalar(x), btScalar(y), btScalar(z)));
		m_bodies[index] = localCreateRigidBody(btScalar(mass), offset*transform, m_shapes[index]);
		m_bodies[index]->setDamping(0.05, 0.85);
		//m_bodies[index]->setDeactivationTime(0.8);
		//m_bodies[index]->setSleepingThresholds(1.6, 2.5);
		break;
		}
		default: {break; }
		}
		(m_bodies[index])->setUserPointer(&(m_ids[index]));
	}

	// CREATE 6-DoF JOINT: 
	void Create6DOF(int jointIndex, int index1, int index2, const btVector3& origin1,const btVector3& origin2, btScalar euler1, btScalar euler2, btScalar euler3, btScalar APLow, btScalar APHigh, btScalar MLLow, btScalar MLHigh)
	{
		btGeneric6DofConstraint * joint;
		btTransform localA, localB;

		localA.setIdentity(); localB.setIdentity();
		//to make the hinge basis oriented in Z direction I used .setEuler rotation method
		localA.getBasis().setEulerZYX(euler1, euler2, euler3); localA.setOrigin(origin1);
		localB.getBasis().setEulerZYX(euler1, euler2, euler3); localB.setOrigin(origin2);

		joint = new btGeneric6DofConstraint(*m_bodies[index1], *m_bodies[index2], localA, localB, false);
		// first 3 - translational DoFs, second 3 - rotational. 4th - rotation around Y-axis (vertical)
		joint->setLimit(0, 0, 0);//the limits of the joint
		joint->setLimit(1, 0, 0);
		joint->setLimit(2, 0, 0);
		//joint->setLimit(3, 0, 0);
		//joint->setLimit(5, btScalar(APLow), btScalar(APHigh));
		//joint->setLimit(4, btScalar(MLLow), btScalar(MLHigh));
		joint->setLimit(3, 0, 0);
		joint->setLimit(4, btScalar(MLLow), btScalar(MLHigh));
		joint->setLimit(5, btScalar(APLow), btScalar(APHigh));
		joint->enableFeedback(true);
		joint->setJointFeedback(&fg);
		m_joints[jointIndex] = joint;
		joint->setDbgDrawSize(CONSTRAINT_DEBUG_SIZE);

		m_ownerWorld->addConstraint(m_joints[jointIndex], true);

	}

	// ACTUATE JOINT:
	void ActuateJoint(int joint_index, int motor_index, double desiredAngle, btScalar timeStep)
	{

		double diff, targetVel;
		float MaxForce = 10.0f;
		diff = desiredAngle - m_joints[joint_index]->getRotationalLimitMotor(motor_index)->m_currentPosition;
		targetVel = diff / timeStep;
		m_joints[joint_index]->getRotationalLimitMotor(motor_index)->m_enableMotor = true;
		m_joints[joint_index]->getRotationalLimitMotor(motor_index)->m_targetVelocity = targetVel;
		m_joints[joint_index]->getRotationalLimitMotor(motor_index)->m_maxMotorForce = MaxForce;
		m_joints[joint_index]->getRotationalLimitMotor(motor_index)->m_maxLimitForce = 25.0f;
	}
	// save one-dimensional vector of neuronal data (requires external counter):
	void save_1dfileN(vector<double> data, string filename, int counter)
	{
		ofstream saveFile;
		saveFile.open(filename, ios_base::app);
		for (unsigned i = 0; i < data.size(); i++)
		{
			saveFile << setprecision(0) << counter << " " << setprecision(6) << data.at(i) << endl;
			counter++;
		}

		saveFile.close();
	}
	// save one-dimensional vector of joint angle data (uses internal counter):
	void save_1dfileJ(vector<double> data, string filename)
	{
		int counter = 1;
		ofstream saveFile;
		saveFile.open(filename, ios_base::app);
		for (unsigned i = 0; i < data.size(); i++)
		{
			saveFile << setprecision(0)<<counter<< " " << setprecision(6) <<data.at(i)<< endl;
			counter++;
		}
		
		saveFile.close();
	}

	void save_1DbtV3(vector<btVector3> data, string filename)
	{
		ofstream saveFile;
		saveFile.open(filename, ios_base::app);
		saveFile << setprecision(4);
		for (unsigned i = 0; i < data.size(); i++)
		{
			saveFile << data.at(i).x() << " " << data.at(i).y() << " " << data.at(i).z() << " " << endl;
		}

		saveFile.close();
	}

	void save_1DInt(vector<int> data, string filename)
	{
		ofstream saveFile;
		saveFile.open(filename, ios_base::app);
		saveFile << setprecision(4);
		for (unsigned i = 0; i < data.size(); i++)
		{
			saveFile << data.at(i) << endl;
		}

		saveFile.close();
	}
	
	void save_2DDbl(vector<vector<double>> data, string filename)
	{
		ofstream saveFile;
		saveFile.open(filename, ios_base::app);
		saveFile << setprecision(4);
		for (unsigned i = 0; i < data.size(); i++)
		{
			for (unsigned j = 0; j < data.at(i).size(); j++)
			{
				saveFile << data.at(i).at(j) << " ";
			}
			saveFile << endl;
		}

		saveFile.close();
	}

	void save_2DInt(vector<vector<int > > data, string filename)
	{
		ofstream saveFile;
		saveFile.open(filename, ios_base::app);
		saveFile << setprecision(4);
		for (unsigned i = 0; i < data[0].size(); i++)
		{
			//cout << "data[0].size() = " << data[0].size() << endl;
			for (unsigned j = 0; j < 2; j++)
			{
				//cout << "data.size() = " << data.size() << endl;
				saveFile << data.at(j).at(i) << " ";
				//cout << data.at(j).at(i) << " ";
			}
			saveFile << endl; //cout << endl;
		}

		saveFile.close();
	}

	void save_2DbtV3(vector<vector<btVector3>> data, string filename)
	{
		ofstream saveFile;
		saveFile.open(filename, ios_base::app);
		saveFile << setprecision(4);
		for (unsigned i = 0; i < data.size(); i++)
		{
			for (unsigned j = 0; j < data.at(i).size(); j++)
			{
				saveFile << data.at(i).at(j).x() << " " << data.at(i).at(j).y() << " " << data.at(i).at(j).z() << " ";
			}
			saveFile << endl;
		}

		saveFile.close();
	}

	//// save one-dimensional vector of any type:
	//template <typename T>
	//void save_1DFile(T data, string filename)
	//{
	//	ofstream saveFile;
	//	saveFile.open(filename, ios_base::app);
	//	cout << "Saving into " << filename << endl;
	//	cout << "Vector size = " << data.size() << ", and sizeof(vector) = " << sizeof(data) << endl;
	//	getchar();
	//	saveFile << setprecision(4);
	//	for (unsigned i = 0; i < data.size(); i++)
	//	{
	//		cout << "Step = " << i << ". Value saved = " << data.at(i) << endl;
	//		cout << "Step = " << i << ". Alt value saved = " << (* data.at(i)) << endl;
	//		saveFile << data.at(i) << endl;
	//		getchar();
	//	}
	//	saveFile.close();
	//}
	//// save two-dimensional vector of any type:
	//template <typename T>
	//void save_2DFile(T data, string filename)
	//{
	//	ofstream saveFile;
	//	saveFile.open(filename, ios_base::app);
	//	saveFile << setprecision(4);
	//	for (unsigned i = 0; i < data.size(); i++)
	//	{
	//		for (unsigned j = 0; j < data.at(i).size(); j++)
	//		{
	//			saveFile << data.at(i).at(j) << " ";
	//		}
	//		
	//		saveFile << endl;
	//	}

	//	saveFile.close();
	//}

	//for saving just one double value into a text file
	void save_1by1_file(double data, string fileName)
	{
		ofstream outputFile;
		outputFile.open(fileName, ios_base::app);
		outputFile << data << endl;
		outputFile.close();
	}

	// GET CoM POSITION of a body part
	btVector3 getCOMposition(int bodyIndex)
	{
		btRigidBody * pointer = m_bodies[bodyIndex];
		btVector3 position = pointer->getCenterOfMassPosition();
		return position;
		
	}
	// Get COM position of the whole body
	btVector3 wholeBodyCOM()
	{
		double thisMass = 0;
		double sumMass = 0;
		btScalar COMcoordX = 0;
		btScalar COMcoordY = 0;
		btScalar COMcoordZ = 0;
		//last "body part" is the platform, which should be discounted
		for (int i = 0; i < BODYPART_COUNT-1; i++)
		{
			btRigidBody * pointer = m_bodies[i];
			btVector3 bodySegCOM = pointer->getCenterOfMassPosition();

			thisMass =  1 / (pointer->getInvMass());
			sumMass += 1 / (pointer->getInvMass());
			COMcoordX += bodySegCOM.x() * thisMass;
			COMcoordY += bodySegCOM.y() * thisMass;
			COMcoordZ += bodySegCOM.z() * thisMass;
		}
		COMcoordX = COMcoordX / sumMass;
		COMcoordY = COMcoordY / sumMass;
		COMcoordZ = COMcoordZ / sumMass;
		btVector3 wholeCOM = btVector3(COMcoordX, COMcoordY, COMcoordZ);
		return wholeCOM;
	}

	double getPelvisHeight()
	{
		double initPelvisHeight = 0.3 + length_foot / 60 + height_leg / 30;
		btRigidBody * p_pelvis = m_bodies[BODYPART_PELVIS];
		btVector3 pelPos = p_pelvis->getCenterOfMassPosition();
		double pelvHeightMember = fabs(initPelvisHeight - pelPos.y());
		//cout << "Pelv. height sensor value = " << pelvHeightMember << endl;
		return pelvHeightMember;
	}

	vector<btVector3> getTargPos()
	{
		double initPelvisHeight = 0.3 + length_foot / 60 + height_leg / 30;
		btScalar leftTargZ = initPelvisHeight / 1.5; //step length = 1/3 of body height, just like in humans
		btScalar leftTargY = 0.3 + length_foot / 120;
		btScalar leftTargX = height_pelvis / 60;//

		btScalar rightTargZ = 0; //step length = 1/3 of body height, just like in humans
		btScalar rightTargY = 0.3 + length_foot / 120;
		btScalar rightTargX = -height_pelvis / 60;//
		vector<btVector3> targPos;
		targPos.push_back(btVector3(leftTargX, leftTargY, leftTargZ));
		targPos.push_back(btVector3(rightTargX, rightTargY, rightTargZ));
		return targPos;
	}

	double onlineFitness(int SimulationStep, int maxStep)
	{
		btRigidBody * p_swingFoot;
		btRigidBody * p_stanceFoot;
		btGeneric6DofConstraint * p_stanceFootAnkleJoint;

		double initPelvisHeight = 0.3 + length_foot / 60 + height_leg / 30;
		double swingTargX;
		double swingTargY = 0.3 + length_foot / 120; // initial feet COM height
		double swingTargZ = initPelvisHeight / 1.5;
		double swingFootInitZ = 0;
		double stanceTargX;
		double stanceTargY = 0.3 + length_foot / 120;
		double stanceTargZ;

		if (SimulationStep < maxStep / 2)
		{
			//cout << "Since SimStep " << SimulationStep << " < maxStep/2 " << maxStep / 2 << ", I am using FitFcn 1 (first half of simulation)." << endl;
			p_swingFoot = m_bodies[BODYPART_LEFT_FOOT];
			p_stanceFoot = m_bodies[BODYPART_RIGHT_FOOT];
			p_stanceFootAnkleJoint = m_joints[JOINT_RIGHT_ANKLE];
			swingTargX = height_pelvis / 60;
			stanceTargX = - height_pelvis / 60;
			stanceTargZ = 0;
		}
		else
		{
			//cout << "Since SimStep " << SimulationStep << " > maxStep/2 " << maxStep / 2 << ", I am using FitFcn 2 (second half of simulation)." << endl;
			p_swingFoot = m_bodies[BODYPART_RIGHT_FOOT];
			p_stanceFoot = m_bodies[BODYPART_LEFT_FOOT];
			p_stanceFootAnkleJoint = m_joints[JOINT_LEFT_ANKLE];
			swingTargX =  - height_pelvis / 60;
			stanceTargX = height_pelvis / 60;
			stanceTargZ = initPelvisHeight / 1.5;
		}

		btRigidBody * p_pelvis = m_bodies[BODYPART_PELVIS];
		
		// Get COM positions:
		btVector3 swingFootPos = p_swingFoot->getCenterOfMassPosition();
		btVector3 stanceFootPos = p_stanceFoot->getCenterOfMassPosition();
		btVector3 pelPos = p_pelvis->getCenterOfMassPosition();

		//get rotation of pelvis around the Y-axis:
		btScalar pelRot, junk1, junk2;
		p_pelvis->getCenterOfMassTransform().getBasis().getEulerZYX(junk1, pelRot, junk2);
		//get joint angles for ankles
//		double stanceFootAnkleAPRot = p_stanceFootAnkleJoint->getRotationalLimitMotor(2)->m_currentPosition;
//		double stanceFootAnkleMLRot = p_stanceFootAnkleJoint->getRotationalLimitMotor(1)->m_currentPosition;

		//values to compare online values of rotation:
		double initPelvRot = 0;
//		double initAnklRotAP = 0;
//		double initAnklRotML = 0;
		
		//Ensure that only forward swing foot movement is rewarded:
		if (swingFootPos.z() < 0) swingFootPos = btVector3(swingFootPos.x(), swingFootPos.y(), btScalar(0));
		if (pelPos.y() > initPelvisHeight) pelPos = btVector3(pelPos.x(), initPelvisHeight, pelPos.z());
		// Fitness members:
		
		double pelvRotMember = 1 / (1 + fabs(initPelvRot - pelRot)); //cout << "pelvRotMember = " << pelvRotMember << endl;
		//cout << "pelvRotMember = 1 / (1 + fabs(" << initPelvRot << " - " << pelRot << ")) = " << pelvRotMember << endl;
																	 //double stanceAnklRotMemberAP = 1 / (1 + abs(initAnklRotAP - stanceFootAnkleAPRot)); //cout << "stanceAnklRotMemberAP = " << stanceAnklRotMemberAP << endl;
		//double stanceAnklRotMemberML = 1 / (1 + abs(initAnklRotML - stanceFootAnkleMLRot)); //cout << "stanceAnklRotMemberML = " << stanceAnklRotMemberML << endl;
		double targetMemberX = 1 / (1 + fabs(swingTargX - swingFootPos.x()));	//cout << "targetMemberX = " << targetMemberX << endl;
		double targetMemberY = 1 / (1 + fabs(swingTargY - swingFootPos.y()));
		double targetMemberZ = 1 / (1 + fabs( (swingTargZ - swingFootPos.z()) / (swingFootPos.z() - swingFootInitZ) )); //cout << "targetMemberZ = " << targetMemberZ << endl;
		double stanceMemberX = 1 / (1 + fabs(stanceTargX - stanceFootPos.x())); //cout << "stanceMemberX = " << stanceMemberX << endl;
		double stanceMemberY = 1 / (1 + fabs(stanceTargY - stanceFootPos.y()));
		double stanceMemberZ = 1 / (1 + fabs(stanceTargZ - stanceFootPos.z())); //cout << "stanceMemberZ = " << stanceMemberZ << endl;
		double pelvHeightMember = 1 / (1 + fabs( (initPelvisHeight - pelPos.y())  )); //cout << "pelvHeightMember = " << pelvHeightMember << endl;
		//double result = pelvRotMember * pelvHeightMember * stanceAnklRotMemberAP * stanceAnklRotMemberML * targetMemberX * targetMemberY * targetMemberZ * stanceMemberX * stanceMemberY * stanceMemberZ;
		// LAST WORKING VERSION:
		// double result = pelvRotMember * pelvHeightMember * targetMemberX * targetMemberY * targetMemberZ * stanceMemberX * stanceMemberY * stanceMemberZ;
		// JUST HEIGHT:
		double result = pelvHeightMember;
		//cout << "swingTargZ = " << swingTargZ << ", swingFootPos.z() = " << swingFootPos.z() << ", swingFootInitZ = " << swingFootInitZ << endl;
		//cout << " Fitness of this SimStep is = " << result << endl;
		//cout << "---------------------------------" << endl;
		
		// //term to penalize end of simulation pelvis displacement:
		// double pelvVelMember;
		// btVector3 pelvLinearVelocity;
		// if (SimulationStep > maxStep*0.75)
		// {
		// 	pelvLinearVelocity = p_pelvis->getLinearVelocity();
		//	//cout << "Step = "<<SimulationStep<<". Pelv velocity X = " << pelvLinearVelocity.x() << " Y = " << pelvLinearVelocity.y() << " Z = " << pelvLinearVelocity.z() << endl;
		//	pelvVelMember = (1 / (1 + fabs(pelvLinearVelocity.x())))*(1 / (1 + fabs(pelvLinearVelocity.y())))*(1 / (1 + fabs(pelvLinearVelocity.z())));
		//	//cout << "PelvVelMember X = " << 1 / (1 + abs(pelvLinearVelocity.x())) << " Y = " << 1 / (1 + abs(pelvLinearVelocity.y())) << " Z = " << 1 / (1 + abs(pelvLinearVelocity.z())) << endl;
		//	//cout << "Step = " << SimulationStep << ". pelvVelMem = " << pelvVelMember << endl;
		//	//cout << "Fit.before = " << result << endl;
		//	result = result * pelvVelMember;
		//	//cout << "Fit. after = " << result << endl;
		// }
		// DEBUG: cout << "SimStep: " << SimulationStep << ", fit = " << result << endl;
		return result;

	}
	
	// detects if the pelvis is fallign below certain height, exits simulation, and saves accumulated fitness up to that simulation step
	void isUpright(double tempFitness, int maxStep, int SimulationStep)
	{
		btRigidBody * pelvis = m_bodies[BODYPART_PELVIS];
		btVector3 pelPos = pelvis->getCenterOfMassPosition();
		double PELV_HEIGHT = 0.3 + footHeight + height_leg / 30;
		double avgTempFitness;

		if (pelPos.y() < PELV_HEIGHT * 0.5)
		{
			avgTempFitness = tempFitness / maxStep;
			//avgTempFitness = tempFitness;
			//save_1by1_file<double>(avgTempFitness, "fit.txt");
			save_1by1_file(avgTempFitness,"fit.txt");
//#ifndef TRAIN
			//cout << "Exit early due to fall. Curr height: " << pelPos.y() << " < 50% of Init height " << PELV_HEIGHT << " = " << PELV_HEIGHT*0.5 << endl;
			//cout << "Sim.Step: " << SimulationStep << " out of " << maxStep << ". Fitness = " << avgTempFitness << endl;
			//getchar();
//#endif
			exit(0);
		}
		return;
	}

	void printHeight()
	{
		cout << "Head = " << height_head/30 << endl;
		cout << "Torso = " << height_torso/30<< endl; // upper + middle torso
		cout << "Pelvis = " << height_pelvis/30 <<endl;
		cout << "Thigh = " << height_thigh/30<<endl;
		cout << "Shank = " << height_shank/30 <<endl;	
		cout << "Foot = " << footHeight <<endl;
		double totalHeight = height_head / 30 + height_torso / 30 + height_pelvis / 30 + height_thigh / 30 + height_shank / 30 + footHeight;
		cout << "Total height = " <<  totalHeight << endl;
		cout << "Stance width 11% of bh should be = " << totalHeight*0.11 << endl;
		cout << "Width btw L and R feet is = " << (height_pelvis / 30) << endl;
		cout << "Feet widths (each) are = " << footWid << endl;
	}

	void keepActive()
	{
		for (unsigned k = 0; k < 8; k++)
   			{
		 			m_bodies[k]->setActivationState(ACTIVE_TAG);
		  	}
		return;
	}
}; //END OF RagDoll CLASS

//// BINARY TOUCH SENSOR SYSTEM:
static RagdollDemo * ragdollDemo; //for processing touches
//// detects if two bodies are in contact and assigns 1's to respective values in touches[number of body parts + 1 for ground]
bool myContactProcessedCallback(btManifoldPoint& cp,
	void * body0, void * body1)
{
	int
		* ID1, * ID2;
	btCollisionObject * o1 = static_cast<btCollisionObject * >(body0);
	btCollisionObject * o2 = static_cast<btCollisionObject * >(body1);
#ifdef TORSO
	int groundID = 7;
#else
#ifndef KNEES
	int groundID = 6;
#else
	int groundID = 8;
#endif
#endif
	ID1 = static_cast<int * >(o1->getUserPointer());
	ID2 = static_cast<int * >(o2->getUserPointer());

	//printf("ID1 = %d, ID2 = %d\n",*ID1,*ID2); //Technical line to check if the registration of collisions works

	ragdollDemo->touches[*ID1] = 1;
	ragdollDemo->touches[*ID2] = 1;
	ragdollDemo->touchPoints[*ID1] = cp.m_positionWorldOnB;
	ragdollDemo->touchPoints[*ID2] = cp.m_positionWorldOnB;
	
	btVector3 normal = cp.m_normalWorldOnB;

	btScalar angleX = normal.angle(btVector3(1, 0, 0));
	btScalar angleY = normal.angle(btVector3(0, 1, 0));
	btScalar angleZ = normal.angle(btVector3(0, 0, 1));

	btScalar impulseX = cp.m_appliedImpulse*cos(angleX);
	btScalar impulseY = cp.m_appliedImpulse*cos(angleY);
	btScalar impulseZ = cp.m_appliedImpulse*cos(angleZ);

	btScalar timeStep = 1.f / 60.f;

	btScalar forceX = impulseX / timeStep;
	btScalar forceY = impulseY / timeStep;
	btScalar forceZ = impulseZ / timeStep;

	ragdollDemo->forces[*ID1] = btVector3(forceX,forceY,forceZ);
	ragdollDemo->forces[*ID2] = btVector3(forceX, forceY, forceZ);
	return false;
}

// READING FROM LOCAL DIRECTORY:
//how to read is directed here:
void load_data(const string& file_name, vector<vector<float > >& data) {
	ifstream is(file_name, ios::in | ios::binary);
	if (!is.is_open()) {
		cout << "Failed open " << file_name << endl;
		return;
	} 
	double i;
	string line;
	while (getline(is, line)) {
		stringstream ss(line);
		data.push_back(vector<float >());
		while (ss >> i)
			data.back().push_back(i);
	}
	is.close();
}

// hardcode the file name here:
RagdollDemo::RagdollDemo() :m_inputFileName("weights.txt")
{}
// execute reading:
void RagdollDemo::initParams(const std::string& inputFileName)
{
	if (!inputFileName.empty())
		m_inputFileName = inputFileName;
}

#ifdef EXPORT
tuple < vector<vector<float > >, vector<vector<int > > > RagdollDemo::stepSNN(vector<float > a, vector<float > b, vector<float > c, vector<float > d, vector<float > v, vector<float > u, vector<vector<float > > w, vector<double > sensor_val, int num_output, int neur_sim_step, int simStep)
#else
vector<vector<float > > RagdollDemo::stepSNN(vector<float > a, vector<float > b, vector<float > c, vector<float > d, vector<float > v, vector<float > u, vector<vector<float > > w, vector<double > sensor_val, int num_output, int neur_sim_step)
#endif
{
	// some housekeeping variables:
	// number of time steps to integrate numerically:
	float timeStep = 2.0f;
	// get total number of neurons:
	int totalNeuronNum = a.size();
	// vector to keep track of inputs to each neuron:
	vector<float > I(a.size()); // has to be generated each simulation step
	int sens_gain = 30;  
	int noise_gain = 5; 
#ifdef HS
	int height_sens_gain = 30;
#endif

	// to generate normal random numbers:
	random_device rd;
	mt19937 gen(rd());
	// values near the mean are the most likely
	// standard deviation affects the dispersion of generated values from the mean
	normal_distribution<> distr(0, 1);

	// 1st row - values of "v", 2nd row - values of "u", 3rd row - sum of spikes generated for each neuron
	vector<vector<float > > output(3,vector<float >(totalNeuronNum));
#ifdef EXPORT
	vector<vector<int > > firings(2, vector<int >());
#endif //EXPORT

#ifdef HS
	int height_sens_neuron;
#endif //end HS

	// MAIN CYCLE:
	for (int t = 0; t < neur_sim_step; t++)
	{
#ifdef HS
		//cout << "Time " << t << ", sensor = " << sensor << endl;
		double die_draw = ((double)rand() / (RAND_MAX));
		// determing if the height sensor neuron fired a spike this neural time step:
		if (die_draw < sensor_val[2])
			height_sens_neuron = 1;
		else
			height_sens_neuron = 0;
#endif

		//generate random thalamic input and find the neurons that have membrane potential > 30 (i.e. that fired):
		// initialize thalamic input:
		for (int i = 0; i < totalNeuronNum; i++)
		{
			// std::cout << "Mem.potential[" << i << "] = " << v[i];
			// sensory input is taken + gaussian noise:
			//I[i] = ((float)(noise_gain * distr(gen))) + sens_gain * sensor_val[0] * w[i][w[i].size()-3] + sens_gain * sensor_val[1] * w[i][w[i].size() - 2] + height_sens_gain * sensor_val[2] * w[i][w[i].size()-1];
			// no noise:
#ifdef HS
			I[i] = sens_gain * sensor_val[0] * w[i][w[i].size() - 3] + sens_gain * sensor_val[1] * w[i][w[i].size() - 2] + height_sens_gain * sensor_val[2] * w[i][w[i].size() - 1];
			//                                         There are two touch sensors: 1st weight is second to last                    2nd weight is the last                          
#else // No height sensor
			I[i] = sens_gain * sensor_val[0] * w[i][w[i].size() - 2] + sens_gain * sensor_val[1] * w[i][w[i].size() - 1];
#endif
			
		}

		// detect fired neurons:
		for (int i = 0; i < totalNeuronNum; i++)
		{
			if (v[i] > 30)
			{
#ifdef EXPORT
				// first row records times when a neuron fires and accounts for the fact that neural network is simulated repeatedly throughout the physical simulation (simStep * simT)
				firings[0].push_back(simStep*neur_sim_step + t + 1);
				// record the neuron number, here as above the values are shifted by 1 for MATLAB (starts counting from 1, not 0):
				firings[1].push_back(i+1);
#endif //EXPORT
				// update the number of spikes fired for current neuron:
				output[2][i] += 1;
				// reset membrane potential of neurons that have just fired:
				v[i] = c[i];
				u[i] = u[i] + d[i];
				// update the influence I vector using the column of S that responds to the fired neuron:
				for (int j = 0; j < totalNeuronNum; j++)
				{
					// signs of neurons are in the first column of the weight mtx 0 - minus, 1 - plus
					// skip update, if the connection from current neuron to target == 0:
					if (w[i][j] > 0)
					{
						I[j] += (w[i][0] - 0.5) / 0.5;
						// sign is extracted from binary:
						// 0 ---> (0 - 0.5)/0.5 = -0.5/0.5 = -1
						// 1 ---> (1 - 0.5)/0.5 = 0.5/0.5 = 1
						// since all meaningful weights are 1s, there is no need to actually multiply sign by weight
					}
				}
			}
		}
		// integrate numerically membrane potential using two steps of 0.5 ms:
		for (int k = 0; k < totalNeuronNum; k++)
		{
			for (int h = 0; h < timeStep; h++)
			{
				v[k] += (float)((1 / timeStep) * (0.04 * pow(v[k], 2) + 5 * v[k] + 140 - u[k] + I[k]));
			}
			// update u:
			u[k] += a[k] * (b[k] * v[k] - u[k]);
		}
	}

	output[0] = v;
	output[1] = u;

#ifdef EXPORT
	return std::make_tuple(output, firings);
#else // no EXPORT:
	return output;
#endif // EXPORT

}

void RagdollDemo::initPhysics()
{
//ADDED:
//	cout << "Inside init physics" << endl;
	// set the RNG seed:
	srand((unsigned)time(NULL));

	ragdollDemo = this;// for processing touches, note spelling "ragdollDemo"
	gContactProcessedCallback = myContactProcessedCallback; //Registers the collision

	// initializing all of CTRNN params here: 
	maxStep = 150; // simulation time
	neur_sim_step = 50; // neural simulation time (per each physics simulation step)
	bodyCount = sizeof(IDs)/sizeof(IDs[0]);
	num_input = 2;
#ifdef TORSO
	num_output = 10;
#else //NO TORSO
#ifndef KNEES // AND NO KNEES EITHER!
	num_output = 8;
#else // with KNEES
	num_output = 12;
#endif // end KNEES
#endif // end TORSO

	SimulationStep = 0; //time step counter to exit after desired # of steps
	tempFitness = 0;
	// 1st row - values of "v", 2nd row - values of "u", 3rd row - firing times of the output neurons
	snn_state.push_back(vector<float >()); snn_state.push_back(vector<float >()); snn_state.push_back(vector<float >());
#ifdef EXPORT
	firings.push_back(vector<int >()); firings.push_back(vector<int >());
#endif //EXPORT
	// initialize all of SNN params:
	for (int i = 0; i < N; i++)
	{
		// Types of neurons from Izhikevich 2004. Fig.1
		//        a         b      c      d        I - preferrable input, not relevant here 
		//pars = [0.02      0.2  -65      6       14; ...    % A regular spiking (RS) although Izhikevich 2003 gives d=8
		//	      0.02      0.2  -50      2        0; ...    % C chattering
		//        0.2       0.26 -65      0        -- ...    % H Class 2 excitable - firing rate directly reflects the input current
		//                                                   % although description in the paper tells that this is Class1. 
		//                                                   % However, model dynamics shows initial increase in fq that later relaxes 
		//                                                   % to a lower value whereas Class 2 remains consistent.  
		//float neuron_rand_param = (float)rand() / (RAND_MAX + 1);
		// to standardize different simulations -> all neurons are Class2:
		a.push_back((float)0.2);
		b.push_back((float)0.26);
		// c.push_back((float)(-65 + 15*sqrt(neuron_rand_param))); // sqrt biases neurons to chattering type, pow - to RS
		// d.push_back((float)8 + 2*sqrt(neuron_rand_param));
		c.push_back((float)(-65));
		d.push_back((float)0);
		//v.push_back((float)(-65));
		snn_state[0].push_back((float)(-65));
		//u.push_back(v[i] * b[i]);
		snn_state[1].push_back(snn_state[0][i] * b[i]);
		// empty vector for firings:
		snn_state[2].push_back(0.0f);
	}


#ifdef TRAIN
	pause = false;// should be false
#else
	pause = true;//should be true
#endif
	oneStep = false;

//Override pause setting if it's a COM file:
#ifdef EXPORT
	pause = false;
#endif // EXPORT

//---- intializing bodypart IDs
	for (int i = 0; i < BODYPART_COUNT+1; i++)
	{
		IDs[i] = i;
	};

#ifdef JOINT
	//intialize 2d vector for joint angles:
	for (int i = 0; i < num_output; i++)
	{
		joint_val.push_back(vector<double>());
		for (int j = 0; j < maxStep; j++)
		{
			joint_val[i].push_back(double());		
		}
	}
#endif

#ifdef COM
	//intialize 2d vector for joint angls and forces [number of joints X number of time steps]:
	for (int i = 0; i < num_output; i++)
	{
		jointAngs.push_back(vector<double>());
		jointForces.push_back(vector<double>());
		for (int j = 0; j < maxStep; j++)
		{
			jointAngs[i].push_back(double());
			jointForces[i].push_back(double());
		}
	}
	//cout << "Sizeof(jointForces) = " << sizeof(jointForces) << ", sizeof(jointForces[0]) = " << sizeof(jointForces[0]) << endl;
	//init COM, feet forces, swing foot touch sensor, and swing foot COM position vectors:
	for (int j = 0; j < maxStep; j++)
	{
		COMpath.push_back(btVector3());
		leftFootForce.push_back(btVector3());
		rightFootForce.push_back(btVector3());
		swingFootTouch.push_back(int());
		swingFootCOMtrace.push_back(btVector3());
	}

#endif


	//READ WEIGHTS:
	// from input file into var called 'w': 
	load_data(m_inputFileName, w);
//END ADDED
	// Setup the basic world
	setTexturing(true);
	setShadows(true);
	setCameraDistance(btScalar(9.));// CAMERA DISTANCE in the beginning of the simulation
	m_collisionConfiguration = new btDefaultCollisionConfiguration();
	m_dispatcher = new btCollisionDispatcher(m_collisionConfiguration);
	btVector3 worldAabbMin(-10000,-10000,-10000);
	btVector3 worldAabbMax(10000,10000,10000);
	m_broadphase = new btAxisSweep3 (worldAabbMin, worldAabbMax);
	m_solver = new btSequentialImpulseConstraintSolver;
	m_dynamicsWorld = new btDiscreteDynamicsWorld(m_dispatcher,m_broadphase,m_solver,m_collisionConfiguration);
	//m_dynamicsWorld->getDispatchInfo().m_useConvexConservativeDistanceUtil = true;
	//m_dynamicsWorld->getDispatchInfo().m_convexConservativeDistanceThreshold = 0.01f;
	// Setup a big ground box
	{
		btCollisionShape* groundShape = new btBoxShape(btVector3(btScalar(200.),btScalar(10.),btScalar(200.)));
		m_collisionShapes.push_back(groundShape);
		btTransform groundTransform;
		groundTransform.setIdentity();
		groundTransform.setOrigin(btVector3(0,-10,0));
//#define CREATE_GROUND_COLLISION_OBJECT = 1;
#ifdef CREATE_GROUND_COLLISION_OBJECT
		btCollisionObject* fixedGround = new btCollisionObject();
		fixedGround->setCollisionShape(groundShape);
		fixedGround->setWorldTransform(groundTransform);
#ifdef TORSO
		fixedGround->setUserPointer(&IDs[7]);
#else
#ifndef KNEES
		fixedGround->setUserPointer(&IDs[6]);
#else
		fixedGround->setUserPointer(&IDs[8]);
#endif
#endif
		m_dynamicsWorld->addCollisionObject(fixedGround);
#else
		localCreateRigidBody(btScalar(0.),groundTransform,groundShape);
#endif //CREATE_GROUND_COLLISION_OBJECT
	}
	//Correct default gravity setting to more correct 9.81 value
	m_dynamicsWorld->setGravity(btVector3(0, -9.81, 0));
	// Spawn one ragdoll
	btVector3 startOffset(1,0.5,0);
	spawnRagdoll(startOffset);

	// Create the second ragdoll at a different location:
	//startOffset.setValue(-1,0.5,0);
	//spawnRagdoll(startOffset);
	clientResetScene();		
}

void RagdollDemo::spawnRagdoll(const btVector3& startOffset)
{
	RagDoll* ragDoll = new RagDoll (m_dynamicsWorld, startOffset, IDs);
	m_ragdolls.push_back(ragDoll);
}	


//Step through simulation without invoking graphics interface:
void RagdollDemo::stepPhysics(float ms)
{
	// vector of target angle values:
	vector<double > targ_angs(num_output);
	//BULLET note: simple dynamics world doesn't handle fixed-time-stepping
	//Roman Popov note: timestep is set to bullet's internal tick = 1/60 of a second. This is done to create constancy between
	//graphics ON/OFF versions of the robot. Actuation timestep is 5 * 1/60 of a second, as it removes movement jitter.

	//Time steps:
	float timeStep = 1.0f / 60.0f;
	float ActuateTimeStep = 5 * timeStep;

	if (m_dynamicsWorld)
	{
		//cout << "Inside m_dynamicsWorld" << endl;
		if (!pause || oneStep)
		{
			//cout << "Inside pause || oneStep" << endl;
			for (int l = 0; l < TICKS_PER_DISPLAY; l++)
			{
				//Intiation of the touches
				for (int i = 0; i < bodyCount; i++)
				{
					touches[i] = 0;
					forces[i] = btVector3(0, 0, 0);
				}
				//Making sure all the body parts are active every time step:
				//Body parts change color when inactive for sometime:
				m_ragdolls[0]->keepActive();
				// Populate sensor_val for the first update of CTRNN:
				if (SimulationStep == 0)
				{
					for (int j = 0; j < num_input; j++)
					{
						sensor_val.push_back(0.0);
					}
#ifdef HS
					// one more time for hte height sens:
					sensor_val.push_back(0.0);
#endif
				}
				//update neronal states:
#ifdef EXPORT
				tie(snn_state, firings) = stepSNN(a, b, c, d, snn_state[0], snn_state[1], w, sensor_val, num_output, neur_sim_step, SimulationStep);
				m_ragdolls[0]->save_2DInt(firings, "firings.txt");
				firings.clear();
#else // if no EXPORT:
				snn_state = stepSNN(a, b, c, d, snn_state[0], snn_state[1], w, sensor_val, num_output, neur_sim_step);
#endif // EXPORT
				// convert spikes into motor commands by estimating the ratio of maximal firing rate:
				// Chattering neuron has circa 15 spikes per 20 ms = 750 spikes per 1000 ms or 750 Hz (estimated using izhikevich matlab code with I=100)
				// RS have circa 5.5 spikes per 20 ms = 275 spikes per 1000 ms
				// Class2 at I=100 - 1250 Hz, at I=22 - 400 Hz (8 per 20 ms), I=32 - 500-550 Hz; I=50 - 15 per 20; 75 - 20 per 20
				// There is a linear regression that fits these data fareely well:
				// Spikes_per_s = 11*Input_current + 166.65
				double total_max_input = 30.0 + 30.0 + 30.0 + N;
				// touch sensor gains 30 + 30, height sensor gain 30, and 50 neurons
				double max_fir_rate = (0.3*total_max_input) * (neur_sim_step / 1000.0); // adjust the max fir.rate to the actual neural sim time
				double forward_effort, backward_effort;
				int motor_count = 0;
				for (int z = 0; z < 2 * num_output; z = z + 2)
				{
					//targ_angs[z] = (double)(2 * snn_state[2][N - z] - 100) / 100; // map [0,99] to [-1,1]
					forward_effort = (snn_state[2][N - (z + 1)] / max_fir_rate);
					backward_effort = (snn_state[2][N - (z + 2)] / max_fir_rate);
					//cout << "Forward = " << forward_effort << ", backward = " << backward_effort << "; [" << N - z << "," << N - z - 1 << "] neurons" << endl;
					targ_angs[motor_count] = (forward_effort - backward_effort);
					if (targ_angs[motor_count] > 1.0)
						targ_angs[motor_count] = 1.0;
					//cout << "Motor #" << z + 1 << " gets value from neuron #" << N - z << " = " << snn_state[2][N - (z+1)] << "/" << max_fir_rate << " = " << targ_angs[z] << endl;
					//cout << "Motor #" << z + 1 << " gets value " << targ_angs[motor_count] << endl;
					motor_count++;
				}
				// for all motors
				for (int i = 0; i < num_output; i++)
				{
					double targetAngle = targ_angs[i];
					switch (i)
					{
					case 0: //Left Hip ML
					{
						targetAngle = ((HIP_ML_H - HIP_ML_L) / 2) * targetAngle + ((HIP_ML_H + HIP_ML_L) / 2); //HIP_ML [-38.8; 30.5]
																											   //cout << "Sending target angle = " << targetAngle * 180 / M_PI << " to " << i << "-th motor that has limits [" << HIP_ML_L * 180 / M_PI << "," << HIP_ML_H * 180 / M_PI << "]" << endl;

						break;
					}
					case 1: //Left Hip AP
					{
						targetAngle = ((HIP_AP_H - HIP_AP_L) / 2) * targetAngle + ((HIP_AP_H + HIP_AP_L) / 2); //HIP_AP [-19; 121]
																											   //cout << "Sending target angle = " << targetAngle * 180 / M_PI << " to " << i << "-th motor that has limits [" << HIP_AP_L * 180 / M_PI << "," << HIP_AP_H * 180 / M_PI << "]" << endl;
						break;
					}
					case 2: //Right Hip ML
					{
						targetAngle = ((HIP_ML_H - HIP_ML_L) / 2) * targetAngle + ((HIP_ML_H + HIP_ML_L) / 2); //HIP_ML [-38.8; 30.5]
																											   //cout << "Sending target angle = " << targetAngle * 180 / M_PI << " to " << i << "-th motor that has limits [" << HIP_ML_L * 180 / M_PI << "," << HIP_ML_H * 180 / M_PI << "]" << endl;
						break;
					}
					case 3: //Right Hip AP
					{
						targetAngle = ((HIP_AP_H - HIP_AP_L) / 2) * targetAngle + ((HIP_AP_H + HIP_AP_L) / 2); //HIP_AP [-19; 121]
																											   //cout << "Sending target angle = " << targetAngle * 180 / M_PI << " to " << i << "-th motor that has limits [" << HIP_AP_L * 180 / M_PI << "," << HIP_AP_H * 180 / M_PI << "]" << endl;
						break;
					}
					case 4: //Left Ankle ML
					{
						targetAngle = ((ANKL_ML_H - ANKL_ML_L) / 2) * targetAngle + ((ANKL_ML_H + ANKL_ML_L) / 2); //ANKL_ML [-27.75; 27.75]
																												   //cout << "Sending target angle = " << targetAngle * 180 / M_PI << " to " << i << "-th motor that has limits [" << ANKL_ML_L * 180 / M_PI << "," << ANKL_ML_H * 180 / M_PI << "]" << endl;
						break;
					}
					case 5: //Left Ankle AP
					{
						targetAngle = ((ANKL_AP_H - ANKL_AP_L) / 2) * targetAngle + ((ANKL_AP_H + ANKL_AP_L) / 2); //ANKL_AP [39.7; -15.3]
																												   //cout << "Sending target angle = " << targetAngle * 180 / M_PI << " to " << i << "-th motor that has limits [" << ANKL_AP_L * 180 / M_PI << "," << ANKL_AP_H * 180 / M_PI << "]" << endl;
						break;
					}
					case 6: //Right Ankle ML
					{
						targetAngle = ((ANKL_ML_H - ANKL_ML_L) / 2) * targetAngle + ((ANKL_ML_H + ANKL_ML_L) / 2); //ANKL_ML [-27.75; 27.75]
						break;
					}
					case 7: //Right Ankle AP
					{
						targetAngle = ((ANKL_AP_H - ANKL_AP_L) / 2) * targetAngle + ((ANKL_AP_H + ANKL_AP_L) / 2); //ANKL_AP [39.7; -15.3]
						break;
					}
#ifdef TORSO
					case 8: // Body-pelvis, ML
					{
						targetAngle = ((TP_ML_H - TP_ML_L) / 2) * targetAngle + ((TP_ML_H + TP_ML_L) / 2); //TP_ML [-25.45; 26.25]
						break;
					}
					case 9: // Body-pelvis AP
					{
						targetAngle = ((TP_AP_H - TP_AP_L) / 2) * targetAngle + ((TP_AP_H + TP_AP_L) / 2); //TP_AP [-57.65; 29.75]															  
						break;
					}
#endif
#ifdef KNEES
					case 8: // Left Knee, ML
					{
						targetAngle = ((KNEE_ML_H - KNEE_ML_L) / 2) * targetAngle + ((KNEE_ML_H + KNEE_ML_L) / 2); //KNEE_ML [0; 0]
						break;
					}
					case 9: // Left Knee AP
					{
						targetAngle = ((KNEE_AP_H - KNEE_AP_L) / 2) * targetAngle + ((KNEE_AP_H + KNEE_AP_L) / 2); //KNEE_AP [-132; 0]
						break;
					}
					case 10: // Right Knee, ML
					{
						targetAngle = ((KNEE_ML_H - KNEE_ML_L) / 2) * targetAngle + ((KNEE_ML_H + KNEE_ML_L) / 2); //KNEE_ML [0; 0]
						break;
					}
					case 11: // Right Knee, AP
					{
						targetAngle = ((KNEE_AP_H - KNEE_AP_L) / 2) * targetAngle + ((KNEE_AP_H + KNEE_AP_L) / 2); //KNEE_AP [-132; 0]
						break;
					}
#endif
					default:
						break;
					}

					m_ragdolls[0]->ActuateJoint(i / 2, fabs(sin(i*M_PI / 2)) + 1, targetAngle, ActuateTimeStep);

				}
				// END UPDATE MOTORS
				m_dynamicsWorld->stepSimulation(timeStep, 0);
#ifdef COM
				btVector3 axisInA(0, 1, 0);
				btScalar appliedImpulse;
				for (int i = 0; i < num_output; i++)
				{
					jointAngs[i][SimulationStep] = m_ragdolls[0]->m_joints[i / 2]->getRotationalLimitMotor(fabs(sin(i*M_PI / 2)) + 1)->m_currentPosition * 180 / M_PI;
					//jointForces[i][SimulationStep] = m_ragdolls[0]->m_joints[i / 2]->getRotationalLimitMotor(abs(sin(i*M_PI / 2)) + 1)->m_accumulatedImpulse / (1.f / 60.f);
					//jointForces[i][SimulationStep] = m_ragdolls[0]->m_joints[i / 2]->getAppliedImpulse() / (1.f / 60.f);


					appliedImpulse = m_ragdolls[0]->m_joints[i / 2]->getJointFeedback()->m_appliedTorqueBodyA.dot(axisInA);
					jointForces[i][SimulationStep] = appliedImpulse / (1.f / 60.f);
					double currAng = m_ragdolls[0]->m_joints[i / 2]->getRotationalLimitMotor(fabs(sin(i*M_PI / 2)) + 1)->m_currentPosition;
					//cout << "SimStep: " << SimulationStep << ", joint num " << (int)i / 2 << ", motor num " << abs(sin(i*M_PI / 2)) + 1 << ". Current pos(deg) = " << currAng * 180 / M_PI << endl;
					COMpath[SimulationStep] = m_ragdolls[0]->wholeBodyCOM();
					leftFootForce[SimulationStep] = forces[BODYPART_LEFT_FOOT];
					rightFootForce[SimulationStep] = forces[BODYPART_RIGHT_FOOT];
					swingFootTouch[SimulationStep] = touches[BODYPART_LEFT_FOOT];
					swingFootCOMtrace[SimulationStep] = m_ragdolls[0]->getCOMposition(BODYPART_LEFT_FOOT);
				}
#endif 
#ifndef COM
#ifndef EXPORT
				// Check if robot is still upright:
				m_ragdolls[0]->isUpright(tempFitness, maxStep, SimulationStep);
#endif
#endif
				//if check is passed, continue the simulation and fitness calculation:;
				// !!! DEBUG, remove later
				m_ragdolls[0]->isUpright(tempFitness, maxStep, SimulationStep);
				// !!! DEBUG
				tempFitness += m_ragdolls[0]->onlineFitness(SimulationStep, maxStep);

#ifdef JOINT
				//store joint angles for exporting:
				for (int i = 0; i < num_output; i++)
				{
					joint_val[i][SimulationStep] = m_ragdolls[0]->m_joints[i / 2]->getRotationalLimitMotor(fabs(sin(i*M_PI / 2)) + 1)->m_currentPosition * 180 / M_PI;
					//cout << "Joint[" << i + 1 << "] = " << joint_val[i][SimulationStep] << endl;
				}
				//DEBUG when touches are updated:
#endif
				//Get new sensor vals:
				sensor_val.clear();
				for (int j = 0; j < num_input; j++)
				{
					sensor_val.push_back(touches[(BODYPART_LEFT_FOOT + j)]);
					//cout << "Sensor val " << j << " from body #" << BODYPART_LEFT_FOOT + j << " = " << touches[(BODYPART_LEFT_FOOT + j)] << endl;
				}
#ifdef HS
				sensor_val.push_back(m_ragdolls[0]->getPelvisHeight());
#endif

#ifdef INFO //prints out some model info:
				if (SimulationStep == 0)
				{
					m_ragdolls[0]->printHeight();
				}
#endif
				// Increase the simulation time step counter:
				SimulationStep++;
// Allows to step through the simulation step-wise
#ifdef STEP
				getchar(); // remove this after debug
#endif

#ifndef TRAIN //if DEMO!!!
						   //Stopping simulation in the end of time for DEMO robots (paused right before the end)
				if (SimulationStep >= maxStep)
				{
#ifdef COM
					cout << "Simstep = " << SimulationStep << endl;
					m_ragdolls[0]->save_1DbtV3(leftFootForce, "leftFootForce.txt");
					m_ragdolls[0]->save_1DbtV3(rightFootForce, "rightFootForce.txt");
					m_ragdolls[0]->save_1DbtV3(COMpath, "com.txt");
					m_ragdolls[0]->save_1DInt(swingFootTouch, "swingFootTouch.txt");
					m_ragdolls[0]->save_1DbtV3(swingFootCOMtrace, "swingFootCOMtrace.txt");
					m_ragdolls[0]->save_2DDbl(jointAngs, "jointAngs.txt");
					m_ragdolls[0]->save_2DDbl(jointForces, "jointForces.txt");

					vector<btVector3> targsPos;
					targsPos.push_back(btVector3()); targsPos.push_back(btVector3());
					targsPos = m_ragdolls[0]->getTargPos();
					m_ragdolls[0]->save_1DbtV3(targsPos, "targets.txt");
#else
					//double fitval = tempFitness / SimulationStep;
					//double fitval = tempFitness;
					//cout << "SimStep: " << SimulationStep << ", C++ fitness: " << fitval << endl;
					//getchar();

#endif
					// //!!! TO DEBUG ONLY. REMOVE FOR EXPORT LATER
					//double fitval = tempFitness / SimulationStep;
					//ofstream outputFile;
					//outputFile.open("fit.txt", ios_base::app);
					//outputFile << fitval << endl;
					//outputFile.close();
					// //!!! TO DEBUG ONLY
					exit(0);
				}
#else // IF TRAIN:
				if (SimulationStep >= maxStep)
				{
					double fitval = tempFitness / SimulationStep;
					//double fitval = tempFitness;
					ofstream outputFile;
					outputFile.open("fit.txt", ios_base::app);
					outputFile << fitval << endl;
					outputFile.close();
					exit(0);
				}
#endif

				// make oneStep false, so that the simulation is paused
				// and waiting next press of the button:
				if (oneStep)
				{
					oneStep = false;
					pause = true;
				}
			}// END NO VIDEO LOOP
		}// END if(!pause && oneStep)

		 //optional but useful: debug drawing
		m_dynamicsWorld->debugDrawWorld();
#ifdef JOINT
		//saving the joint angle values to a file:
		if (SimulationStep >= maxStep)
		{
			for (int i = 0; i < num_output; i++)
			{
				string fileName;
				fileName = "joint" + to_string(i + 1) + ".txt";
				m_ragdolls[0]->save_1dfileJ(joint_val[i], fileName); //-> to be used if only end of simulation fitness is reported
			}
			exit(0);
		}
#endif
	}

	//vector<double > targ_angs;
//	// motor acruating timeStep
//	float ActuateTimeStep = 5*ms;
//
//	if (m_dynamicsWorld)
//	{		
////		cout << "Inside m_dynamicsWorld" << endl;
//		if (!pause || oneStep)
//		{
////			cout << "Inside pause || oneStep" << endl;
//			//Intiation of the touches
//			for (int i = 0; i < bodyCount; i++)
//			{
//				touches[i] = 0;
//				forces[i] = btVector3(0, 0, 0);
//			}
//
////			cout << "Touches initialized" << endl;
//			//Making sure all the body parts are active every time step:
//			//Body parts change color when inactive for sometime:
////			m_ragdolls[0]->keepActive();
//
////			cout << "Bodies are activated" << endl;
//			// Populate sensor_val for the first update of CTRNN:
//			if (SimulationStep == 0)
//			{
//				for (int j = 0; j < num_input; j++)
//				{
//					sensor_val.push_back(0);
//				}
//			}
////			cout << "Sensor values initialized" << endl;
//
//#ifdef NEURON
//			vector<vector<vector<double>>> temp3Darray;
//			temp3Darray = eulerEXPORT(neural_step, h, tau, w, neuron_val, bias, sensor_val, gain);
//
//			for (int step = 0; step < (neural_step / h); step++)
//			{
//				//cout << "Sim. step = " << SimulationStep << ", int.neuronal tick = " << step << endl;
//				for (int layr = 0; layr < neuronHist[0].size(); layr++)
//				{
//					//cout << "neuronHist. Layer " << layr << ": ";
//					for (int nrn = 0; nrn < neuronHist[0][layr].size(); nrn++)
//					{
//						neuronHist[SimulationStep*(neural_step / h) + step][layr][nrn] = temp3Darray[step][layr][nrn];
//						//cout << "neuronHist["<<(SimulationStep*(neural_step / h) + step)<<"]["<<layr<<"]["<<nrn<<"] = "<< (neuronHist[SimulationStep*(neural_step / h) + step][layr][nrn]) <<"  = temp3Darray["<<step<<"]["<<layr<<"]["<<nrn<<"] = " << temp3Darray[step][layr][nrn] << endl;
//					}
//					//cout << endl;
//				}
//			}
//#endif//NEURON
//
//			//update neronal states:
//			//neuron_val = euler(neural_step, h, tau, w, neuron_val, bias, sensor_val, gain);
//			outFired = stepSNN(a, b, c, d, v, u, w, sensor_val, num_output);
//
//			//extract values from output layer:
//			for (int z = 0; z < num_output; z++)
//			{
//				targ_angs[z] = (2*outFired[z]-100) / 100; // map [0,99] to [-1,1]
//			}
//			
//				
////			cout << "Robot initialized and conditioned. Ready to compute target angles" << endl;
//
//			// for all motors
//			for (int i = 0; i < num_output; i++)
//			{
//				//double targetAngle = tanh((targ_angs.at(i) + bias[2][i]) * gain[2][i]);
//				double targetAngle = targ_angs[i];
//				//activityIndex += abs(targetAngle); // <- simplified version. Should be: targetAngle / [range of all possible values for targetAngle]
//				// but since [range] = 2, or 1 on each side (it's symmetrical), then it is just targetAngle. The larger the angle, the higher the activity index. 
//				// check which motor is to be actuated (values are scaled to the desired angle)
//				//cout << "SimStep: " << SimulationStep << ". targetAngle(unscaled) = " << targetAngle << ". Accumulated activityIndex = " << activityIndex << endl;
//				switch (i)
//				{
//				case 0: //Left Hip ML
//				{
//					targetAngle = ((HIP_ML_H - HIP_ML_L) / 2) * targetAngle + ((HIP_ML_H + HIP_ML_L) / 2); //HIP_ML [-38.8; 30.5]
//					//cout << "Sending target angle = " << targetAngle * 180 / M_PI << " to " << i << "-th motor that has limits [" << HIP_ML_L * 180 / M_PI << "," << HIP_ML_H * 180 / M_PI << "]" << endl;
//					break;
//				}
//				case 1: //Left Hip AP
//				{
//					targetAngle = ((HIP_AP_H - HIP_AP_L) / 2) * targetAngle + ((HIP_AP_H + HIP_AP_L) / 2); //HIP_AP [-19; 121]
//					//targetAngle = 67.5 * targetAngle - 45; //[22.5; -112.5] according to constraints on the joints (?)
//					break;
//				}
//				case 2: //Right Hip ML
//				{
//					targetAngle = ((HIP_ML_H - HIP_ML_L) / 2) * targetAngle + ((HIP_ML_H + HIP_ML_L) / 2); //HIP_ML [-38.8; 30.5]
//					break;
//				}
//				case 3: //Right Hip AP
//				{
//					targetAngle = ((HIP_AP_H - HIP_AP_L) / 2) * targetAngle + ((HIP_AP_H + HIP_AP_L) / 2); //HIP_AP [-19; 121]
//					//targetAngle = 67.5 * targetAngle - 45; //[22.5; -112.5] according to constraints on the joints (?)
//					break;
//				}
//				case 4: //Left Ankle ML
//				{
//					targetAngle = ((ANKL_ML_H - ANKL_ML_L) / 2) * targetAngle + ((ANKL_ML_H + ANKL_ML_L) / 2); //ANKL_ML [-27.75; 27.75]
//					break;
//				}
//				case 5: //Left Ankle AP
//				{
//					targetAngle = ((ANKL_AP_H - ANKL_AP_L) / 2) * targetAngle + ((ANKL_AP_H + ANKL_AP_L) / 2); //ANKL_AP [39.7; -15.3]
//					//targetAngle = 36 * targetAngle - 22.5;//[-58.5; 13.5] according to constraints on the joints (?)
//					break;
//				}
//				case 6: //Right Ankle ML
//				{
//					targetAngle = ((ANKL_ML_H - ANKL_ML_L) / 2) * targetAngle + ((ANKL_ML_H + ANKL_ML_L) / 2); //ANKL_ML [-27.75; 27.75]
//					break;
//				}
//				case 7: //Right Ankle AP
//				{
//					targetAngle = ((ANKL_AP_H - ANKL_AP_L) / 2) * targetAngle + ((ANKL_AP_H + ANKL_AP_L) / 2); //ANKL_AP [39.7; -15.3]
//					break;
//				}
//#ifdef TORSO
//				case 8: // Body-pelvis, ML
//				{
//					targetAngle = ((TP_ML_H - TP_ML_L) / 2) * targetAngle + ((TP_ML_H + TP_ML_L) / 2); //TP_ML [-25.45; 26.25]
//					break;
//				}
//				case 9: // Body-pelvis AP
//				{
//					targetAngle = ((TP_AP_H - TP_AP_L) / 2) * targetAngle + ((TP_AP_H + TP_AP_L) / 2); //TP_AP [-57.65; 29.75]															  
//					break;
//				}
//#endif
//#ifdef KNEES
//				case 8: // Left Knee, ML
//				{
//					targetAngle = ((KNEE_ML_H - KNEE_ML_L) / 2) * targetAngle + ((KNEE_ML_H + KNEE_ML_L) / 2); //KNEE_ML [0; 0]
//					break;
//				}
//				case 9: // Left Knee AP
//				{
//					targetAngle = ((KNEE_AP_H - KNEE_AP_L) / 2) * targetAngle + ((KNEE_AP_H + KNEE_AP_L) / 2); //KNEE_AP [-132; 0]
//					break;
//				}
//				case 10: // Right Knee, ML
//				{
//					targetAngle = ((KNEE_ML_H - KNEE_ML_L) / 2) * targetAngle + ((KNEE_ML_H + KNEE_ML_L) / 2); //KNEE_ML [0; 0]
//					break;
//				}
//				case 11: // Right Knee, AP
//				{
//					targetAngle = ((KNEE_AP_H - KNEE_AP_L) / 2) * targetAngle + ((KNEE_AP_H + KNEE_AP_L) / 2); //KNEE_AP [-132; 0]
//					break;
//				}
//#endif
//				default:
//					break;
//				}
//
//				m_ragdolls[0]->ActuateJoint(i / 2, fabs(sin(i*M_PI / 2)) + 1, targetAngle, ActuateTimeStep);
//				//double currAng = m_ragdolls[0]->m_joints[i / 2]->getRotationalLimitMotor(abs(sin(i*M_PI / 2)) + 1)->m_currentPosition;
//				//cout << "SS: " << SimulationStep << ", J# " << (int)i / 2 << ", M#" << abs(sin(i*M_PI / 2)) + 1 << ". Current pos(deg) = " << currAng * 180 / M_PI << ". Target given = " << targetAngle * 180 / M_PI << endl;
//
//			}
//			// END UPDATE MOTORS
//
//			m_dynamicsWorld->stepSimulation(ms, 0);
//
//			//cout << "World was updated. New joint values!" << endl;
//#ifdef COM
//			btVector3 axisInA(0, 1, 0);
//			btScalar appliedImpulse;
//			for (int i = 0; i < num_output; i++)
//			{
//				jointAngs[i][SimulationStep] = m_ragdolls[0]->m_joints[i / 2]->getRotationalLimitMotor(fabs(sin(i*M_PI / 2)) + 1)->m_currentPosition * 180 / M_PI;
//				//jointForces[i][SimulationStep] = m_ragdolls[0]->m_joints[i / 2]->getRotationalLimitMotor(abs(sin(i*M_PI / 2)) + 1)->m_accumulatedImpulse / (1.f / 60.f);
//				//jointForces[i][SimulationStep] = m_ragdolls[0]->m_joints[i / 2]->getAppliedImpulse() / (1.f / 60.f);
//
//
//				appliedImpulse = m_ragdolls[0]->m_joints[i / 2]->getJointFeedback()->m_appliedTorqueBodyA.dot(axisInA);
//				jointForces[i][SimulationStep] = appliedImpulse / (1.f / 60.f);
//				double currAng = m_ragdolls[0]->m_joints[i / 2]->getRotationalLimitMotor(fabs(sin(i*M_PI / 2)) + 1)->m_currentPosition;
//				//cout << "SimStep: " << SimulationStep << ", joint num " << (int)i / 2 << ", motor num " << abs(sin(i*M_PI / 2)) + 1 << ". Current pos(deg) = " << currAng * 180 / M_PI << endl;
//				COMpath[SimulationStep] = m_ragdolls[0]->wholeBodyCOM();
//				leftFootForce[SimulationStep] = forces[BODYPART_LEFT_FOOT];
//				rightFootForce[SimulationStep] = forces[BODYPART_RIGHT_FOOT];
//				swingFootTouch[SimulationStep] = touches[BODYPART_LEFT_FOOT];
//				swingFootCOMtrace[SimulationStep] = m_ragdolls[0]->getCOMposition(BODYPART_LEFT_FOOT);
//		    }
//#endif// COM 
//
//#ifndef COM
//#ifndef NEURON
//			// Check if robot is still upright:
//			m_ragdolls[0]->isUpright(tempFitness, maxStep, SimulationStep);
//#endif//NEURON
//#endif//COM
//			//if check is passed, continue the simulation and fitness calculation:;
//			tempFitness += m_ragdolls[0]->onlineFitness(SimulationStep, maxStep);
//			//cout << "Simstep = " << SimulationStep << ". Fitness = " << tempFitness << endl;
//
//
//#ifdef JOINT
//			//store joint angles for exporting:
//			for (int i = 0; i < num_output; i++)
//			{
//				joint_val[i][SimulationStep] = m_ragdolls[0]->m_joints[i/2]->getRotationalLimitMotor(fabs(sin(i*M_PI / 2)) + 1)->m_currentPosition*180/M_PI;
//				//cout << "Joint[" << i + 1 << "] = " << joint_val[i][SimulationStep] << endl;
//			}
//			//DEBUG when touches are updated:
//#endif //JOINT
//
//			//Get new sensor vals:
//			sensor_val.clear();
//			for (int j = 0; j < num_input; j++)
//			{
//				sensor_val.push_back(touches[(BODYPART_LEFT_FOOT + j)]);
//				//cout << "Sensor val " << j << " from body #" << BODYPART_LEFT_FOOT + j << " = " << touches[(BODYPART_LEFT_FOOT + j)] << endl;
//			}
//
//#ifdef INFO //prints out some model info:
//			if (SimulationStep == 0)
//			{
//				m_ragdolls[0]->printHeight();
//			}
//#endif //INFO
//
//			//cout << "Simulation step = " << SimulationStep << endl;
//			//float ms = getDeltaTimeMicroseconds();
//			//cout << "time delta = " << ms / 1000.0f << " ms." << endl;
//			// Increase the simulation time step counter:
//			SimulationStep++;
//
////#ifdef TAU_SWITCH
////			if (SimulationStep == (maxStep / 2))
////			{
////				cout << "Flipping taus..." << endl;
////				tau = flipTau(tau);
////			}
////#endif //TAU_SWITCH
//
//#ifndef TRAIN //if DEMO!!!
//
//			//Stopping simulation in the end of time for DEMO robots (paused right before the end)
//			if (SimulationStep >= maxStep)
//			{
//#ifdef COM
//				cout << "Simstep = " << SimulationStep << endl;
//				m_ragdolls[0]->save_1DbtV3(leftFootForce, "leftFootForce.txt");
//				m_ragdolls[0]->save_1DbtV3(rightFootForce, "rightFootForce.txt");
//				m_ragdolls[0]->save_1DbtV3(COMpath, "com.txt");
//				m_ragdolls[0]->save_1DInt(swingFootTouch, "swingFootTouch.txt");
//				m_ragdolls[0]->save_1DbtV3(swingFootCOMtrace, "swingFootCOMtrace.txt");
//				m_ragdolls[0]->save_2DDbl(jointAngs, "jointAngs.txt");
//				m_ragdolls[0]->save_2DDbl(jointForces, "jointForces.txt");
//
//				vector<btVector3> targsPos;
//				targsPos.push_back(btVector3()); targsPos.push_back(btVector3());
//				targsPos = m_ragdolls[0]->getTargPos();
//				m_ragdolls[0]->save_1DbtV3(targsPos, "targets.txt");
//
//#else // COM
//
//				//double fitval = tempFitness / SimulationStep;
//				double fitval = tempFitness;
//				cout << "SimStep: " << SimulationStep << ", C++ fitness: " << fitval << endl;
//				getchar();
//					
//#endif // COM
//
//#ifdef NEURON
//				//transform 3D to 2D neuronal states vector [time X layer X neuron] -> [time X neuron]
//				vector<vector<double>> outputNeurStates;
//				int layr, neuronCount;
//				for (int t = 0; t < neuronHist.size(); t++)
//				{
//					outputNeurStates.push_back(vector<double>());
//					for (int nrn = 0; nrn < (num_input + num_hidden + num_output); nrn++)
//					{
//						outputNeurStates[t].push_back(double());
//						if (nrn < num_input) { layr = 0; neuronCount = nrn; }
//						if ((nrn < (num_input + num_hidden)) && (nrn >= num_input)) { layr = 1; neuronCount = nrn - num_input; }
//						if (nrn >= (num_input + num_hidden)) { layr = 2; neuronCount = nrn - num_input - num_hidden; }
//						outputNeurStates[t][nrn] = neuronHist[t][layr][neuronCount];
//					}
//				}
//				//export 2d vector:
//				m_ragdolls[0]->save_2DDbl(outputNeurStates, "neuron.txt");
//#endif //NEURON
//				exit(0);
//			} // END IF SIMSTEP >= maxStep
//#else // IF TRAIN:
//			if (SimulationStep >= maxStep)
//			{
//				//double fitval = tempFitness / SimulationStep;
//				double fitval = tempFitness;
//				ofstream outputFile;
//				outputFile.open("fit.txt", ios_base::app);
//				outputFile << fitval << endl;
//				outputFile.close();
//				getchar();
//				exit(0);
//			}
//#endif //END TRAIN
//				
//			// make oneStep false, so that the simulation is paused
//			// and waiting next press of the button:
//			if (oneStep)
//			{
//				oneStep = false;
//				pause = true;
//			}
//		}// END if(!pause && oneStep)
//
//#ifdef JOINT
//		//saving the joint angle values to a file:
//		if (SimulationStep >= maxStep)
//		{
//			for (int i = 0; i < num_output; i++)
//			{
//			string fileName;
//			fileName = "joint" + to_string(i + 1) + ".txt";
//			m_ragdolls[0]->save_1dfileJ(joint_val[i], fileName); //-> to be used if only end of simulation fitness is reported
//			}
//			exit(0);
//		}
//#endif //JOINT
//
//
//	}//End if (m_dynamicsWorld)
//
}//end stepPhysics



void RagdollDemo::clientMoveAndDisplay()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	//cout << "Inside ClientMoveAndDisplay" << endl;
	// vector of target angle values:
	vector<double > targ_angs(num_output);
	//BULLET note: simple dynamics world doesn't handle fixed-time-stepping
	//Roman Popov note: timestep is set to bullet's internal tick = 1/60 of a second. This is done to create constancy between
	//graphics ON/OFF versions of the robot. Actuation timestep is 5 * 1/60 of a second, as it removes movement jitter.

	//Time steps:
	float timeStep = 1.0f / 60.0f;
	float ActuateTimeStep = 5*timeStep;

	if (m_dynamicsWorld)
	{		
		//cout << "Inside m_dynamicsWorld" << endl;
		if (!pause || oneStep)
		{
			//cout << "Inside pause || oneStep" << endl;
			for (int l = 0; l < TICKS_PER_DISPLAY; l++)
			{
				//Intiation of the touches
				for (int i = 0; i < bodyCount; i++)
				{
					touches[i] = 0;
					forces[i] = btVector3(0, 0, 0);
				}
				//Making sure all the body parts are active every time step:
				//Body parts change color when inactive for sometime:
				m_ragdolls[0]->keepActive();
				// Populate sensor_val for the first update of CTRNN:
				if (SimulationStep == 0)
				{
					for (int j = 0; j < num_input; j++)
					{
						sensor_val.push_back(0.0);
					}
#ifdef HS
					// one more time for hte height sens:
					sensor_val.push_back(0.0);
#endif
				}
				//update neronal states:
#ifdef EXPORT
				tie(snn_state, firings) = stepSNN(a, b, c, d, snn_state[0], snn_state[1], w, sensor_val, num_output, neur_sim_step,SimulationStep);
				m_ragdolls[0]->save_2DInt(firings, "firings.txt");
				firings.clear();				
#else // if no EXPORT:
				snn_state = stepSNN(a, b, c, d, snn_state[0], snn_state[1], w, sensor_val, neur_sim_step, num_output);
#endif // EXPORT
				// convert spikes into motor commands by estimating the ratio of maximal firing rate:
				// Chattering neuron has circa 15 spikes per 20 ms = 750 spikes per 1000 ms or 750 Hz (estimated using izhikevich matlab code with I=100)
				// RS have circa 5.5 spikes per 20 ms = 275 spikes per 1000 ms
				// Class2 at I=100 - 1250 Hz, at I=22 - 400 Hz (8 per 20 ms), I=32 - 500-550 Hz
				// There is a linear regression that fits these data fareely well:
				// Spikes_per_s = 11*Input_current + 166.65
				double total_max_input = 30.0 + 30.0 + 30.0 + N;
				// touch sensor gains 30 + 30, height sensor gain 30, and 50 neurons
				double max_fir_rate = (0.3*total_max_input) * (neur_sim_step / 1000.0); // adjust the max fir.rate to the actual neural sim time
				double forward_effort, backward_effort;
				int motor_count = 0;
				for (int z = 0; z < 2*num_output; z=z+2)
				{
					//targ_angs[z] = (double)(2 * snn_state[2][N - z] - 100) / 100; // map [0,99] to [-1,1]
					forward_effort = (snn_state[2][N - (z + 1)] / max_fir_rate);
					backward_effort = (snn_state[2][N - (z + 2)] / max_fir_rate);
					//cout << "Forward = " << forward_effort << ", backward = " << backward_effort << "; [" << N - z << "," << N - z - 1 << "] neurons" << endl;
					targ_angs[motor_count] = (forward_effort - backward_effort);
					if (targ_angs[motor_count] > 1.0)
						targ_angs[motor_count] = 1.0;
					//cout << "Motor #" << z + 1 << " gets value from neuron #" << N - z << " = " << snn_state[2][N - (z+1)] << "/" << max_fir_rate << " = " << targ_angs[z] << endl;
					//cout << "Motor #" << z + 1 << " gets value " << targ_angs[motor_count] << endl;
					motor_count++;
				}
				// for all motors
				for (int i = 0; i < num_output; i++)
				{
					double targetAngle = targ_angs[i];
					switch (i)
					{
					case 0: //Left Hip ML
					{
						targetAngle = ((HIP_ML_H - HIP_ML_L) / 2) * targetAngle + ((HIP_ML_H + HIP_ML_L) / 2); //HIP_ML [-38.8; 30.5]
						//cout << "Sending target angle = " << targetAngle * 180 / M_PI << " to " << i << "-th motor that has limits [" << HIP_ML_L * 180 / M_PI << "," << HIP_ML_H * 180 / M_PI << "]" << endl;

						break;
					}
					case 1: //Left Hip AP
					{
						targetAngle = ((HIP_AP_H - HIP_AP_L) / 2) * targetAngle + ((HIP_AP_H + HIP_AP_L) / 2); //HIP_AP [-19; 121]
						//cout << "Sending target angle = " << targetAngle * 180 / M_PI << " to " << i << "-th motor that has limits [" << HIP_AP_L * 180 / M_PI << "," << HIP_AP_H * 180 / M_PI << "]" << endl;
						break;
					}
					case 2: //Right Hip ML
					{
						targetAngle = ((HIP_ML_H - HIP_ML_L) / 2) * targetAngle + ((HIP_ML_H + HIP_ML_L) / 2); //HIP_ML [-38.8; 30.5]
						//cout << "Sending target angle = " << targetAngle * 180 / M_PI << " to " << i << "-th motor that has limits [" << HIP_ML_L * 180 / M_PI << "," << HIP_ML_H * 180 / M_PI << "]" << endl;
						break;
					}
					case 3: //Right Hip AP
					{
						targetAngle = ((HIP_AP_H - HIP_AP_L) / 2) * targetAngle + ((HIP_AP_H + HIP_AP_L) / 2); //HIP_AP [-19; 121]
						//cout << "Sending target angle = " << targetAngle * 180 / M_PI << " to " << i << "-th motor that has limits [" << HIP_AP_L * 180 / M_PI << "," << HIP_AP_H * 180 / M_PI << "]" << endl;
						break;
					}
					case 4: //Left Ankle ML
					{
						targetAngle = ((ANKL_ML_H - ANKL_ML_L) / 2) * targetAngle + ((ANKL_ML_H + ANKL_ML_L) / 2); //ANKL_ML [-27.75; 27.75]
						//cout << "Sending target angle = " << targetAngle * 180 / M_PI << " to " << i << "-th motor that has limits [" << ANKL_ML_L * 180 / M_PI << "," << ANKL_ML_H * 180 / M_PI << "]" << endl;
						break;
					}
					case 5: //Left Ankle AP
					{
						targetAngle = ((ANKL_AP_H - ANKL_AP_L) / 2) * targetAngle + ((ANKL_AP_H + ANKL_AP_L) / 2); //ANKL_AP [39.7; -15.3]
						//cout << "Sending target angle = " << targetAngle * 180 / M_PI << " to " << i << "-th motor that has limits [" << ANKL_AP_L * 180 / M_PI << "," << ANKL_AP_H * 180 / M_PI << "]" << endl;
						break;
					}
					case 6: //Right Ankle ML
					{
						targetAngle = ((ANKL_ML_H - ANKL_ML_L) / 2) * targetAngle + ((ANKL_ML_H + ANKL_ML_L) / 2); //ANKL_ML [-27.75; 27.75]
						break;
					}
					case 7: //Right Ankle AP
					{
						targetAngle = ((ANKL_AP_H - ANKL_AP_L) / 2) * targetAngle + ((ANKL_AP_H + ANKL_AP_L) / 2); //ANKL_AP [39.7; -15.3]
						break;
					}
#ifdef TORSO
					case 8: // Body-pelvis, ML
					{
						targetAngle = ((TP_ML_H - TP_ML_L) / 2) * targetAngle + ((TP_ML_H + TP_ML_L) / 2); //TP_ML [-25.45; 26.25]
						break;
					}
					case 9: // Body-pelvis AP
					{
						targetAngle = ((TP_AP_H - TP_AP_L) / 2) * targetAngle + ((TP_AP_H + TP_AP_L) / 2); //TP_AP [-57.65; 29.75]															  
						break;
					}
#endif
#ifdef KNEES
					case 8: // Left Knee, ML
					{
						targetAngle = ((KNEE_ML_H - KNEE_ML_L) / 2) * targetAngle + ((KNEE_ML_H + KNEE_ML_L) / 2); //KNEE_ML [0; 0]
						break;
					}
					case 9: // Left Knee AP
					{
						targetAngle = ((KNEE_AP_H - KNEE_AP_L) / 2) * targetAngle + ((KNEE_AP_H + KNEE_AP_L) / 2); //KNEE_AP [-132; 0]
						break;
					}
					case 10: // Right Knee, ML
					{
						targetAngle = ((KNEE_ML_H - KNEE_ML_L) / 2) * targetAngle + ((KNEE_ML_H + KNEE_ML_L) / 2); //KNEE_ML [0; 0]
						break;
					}
					case 11: // Right Knee, AP
					{
						targetAngle = ((KNEE_AP_H - KNEE_AP_L) / 2) * targetAngle + ((KNEE_AP_H + KNEE_AP_L) / 2); //KNEE_AP [-132; 0]
						break;
					}
#endif
					default:
						break;
					}

					m_ragdolls[0]->ActuateJoint(i / 2, fabs(sin(i*M_PI / 2)) + 1, targetAngle, ActuateTimeStep);

				}
				// END UPDATE MOTORS
				m_dynamicsWorld->stepSimulation(timeStep, 0);
#ifdef COM
				btVector3 axisInA(0, 1, 0);
				btScalar appliedImpulse;
				for (int i = 0; i < num_output; i++)
				{
					jointAngs[i][SimulationStep] = m_ragdolls[0]->m_joints[i / 2]->getRotationalLimitMotor(fabs(sin(i*M_PI / 2)) + 1)->m_currentPosition * 180 / M_PI;
					//jointForces[i][SimulationStep] = m_ragdolls[0]->m_joints[i / 2]->getRotationalLimitMotor(abs(sin(i*M_PI / 2)) + 1)->m_accumulatedImpulse / (1.f / 60.f);
					//jointForces[i][SimulationStep] = m_ragdolls[0]->m_joints[i / 2]->getAppliedImpulse() / (1.f / 60.f);


					appliedImpulse = m_ragdolls[0]->m_joints[i / 2]->getJointFeedback()->m_appliedTorqueBodyA.dot(axisInA);
					jointForces[i][SimulationStep] = appliedImpulse / (1.f / 60.f);
					double currAng = m_ragdolls[0]->m_joints[i / 2]->getRotationalLimitMotor(fabs(sin(i*M_PI / 2)) + 1)->m_currentPosition;
					//cout << "SimStep: " << SimulationStep << ", joint num " << (int)i / 2 << ", motor num " << abs(sin(i*M_PI / 2)) + 1 << ". Current pos(deg) = " << currAng * 180 / M_PI << endl;
					COMpath[SimulationStep] = m_ragdolls[0]->wholeBodyCOM();
					leftFootForce[SimulationStep] = forces[BODYPART_LEFT_FOOT];
					rightFootForce[SimulationStep] = forces[BODYPART_RIGHT_FOOT];
					swingFootTouch[SimulationStep] = touches[BODYPART_LEFT_FOOT];
					swingFootCOMtrace[SimulationStep] = m_ragdolls[0]->getCOMposition(BODYPART_LEFT_FOOT);
			    }
#endif 
				// Check if robot is still upright:
				m_ragdolls[0]->isUpright(tempFitness, maxStep, SimulationStep);
				//if check is passed, continue the simulation and fitness calculation:;
				tempFitness += m_ragdolls[0]->onlineFitness(SimulationStep, maxStep);

#ifdef JOINT
				//store joint angles for exporting:
				for (int i = 0; i < num_output; i++)
				{
					joint_val[i][SimulationStep] = m_ragdolls[0]->m_joints[i/2]->getRotationalLimitMotor(fabs(sin(i*M_PI / 2)) + 1)->m_currentPosition*180/M_PI;
					//cout << "Joint[" << i + 1 << "] = " << joint_val[i][SimulationStep] << endl;
				}
				//DEBUG when touches are updated:
#endif
				//Get new sensor vals:
				sensor_val.clear();
				for (int j = 0; j < num_input; j++)
				{
					sensor_val.push_back(touches[(BODYPART_LEFT_FOOT + j)]);
					//cout << "Sensor val " << j << " from body #" << BODYPART_LEFT_FOOT + j << " = " << touches[(BODYPART_LEFT_FOOT + j)] << endl;
				}
#ifdef HS
				sensor_val.push_back(m_ragdolls[0]->getPelvisHeight());
#endif

#ifdef INFO //prints out some model info:
				if (SimulationStep == 0)
				{
					m_ragdolls[0]->printHeight();
				}
#endif
				// Increase the simulation time step counter:
				SimulationStep++;

#ifndef TRAIN //if DEMO!!!
				//Stopping simulation in the end of time for DEMO robots (paused right before the end)
				if (SimulationStep >= maxStep)
				{
#ifdef COM
					cout << "Simstep = " << SimulationStep << endl;
					m_ragdolls[0]->save_1DbtV3(leftFootForce, "leftFootForce.txt");
					m_ragdolls[0]->save_1DbtV3(rightFootForce, "rightFootForce.txt");
					m_ragdolls[0]->save_1DbtV3(COMpath, "com.txt");
					m_ragdolls[0]->save_1DInt(swingFootTouch, "swingFootTouch.txt");
					m_ragdolls[0]->save_1DbtV3(swingFootCOMtrace, "swingFootCOMtrace.txt");
					m_ragdolls[0]->save_2DDbl(jointAngs, "jointAngs.txt");
					m_ragdolls[0]->save_2DDbl(jointForces, "jointForces.txt");

					vector<btVector3> targsPos;
					targsPos.push_back(btVector3()); targsPos.push_back(btVector3());
					targsPos = m_ragdolls[0]->getTargPos();
					m_ragdolls[0]->save_1DbtV3(targsPos, "targets.txt");
#else
					//double fitval = tempFitness / SimulationStep;
					//double fitval = tempFitness;
					//cout << "SimStep: " << SimulationStep << ", C++ fitness: " << fitval << endl;
					//getchar();
					
#endif
					//!!! TO DEBUG ONLY. REMOVE FOR EXPORT LATER
					double fitval = tempFitness / SimulationStep;
					ofstream outputFile;
					outputFile.open("fit.txt", ios_base::app);
					outputFile << fitval << endl;
					outputFile.close();
					//!!! TO DEBUG ONLY
					exit(0);
				}
#else // IF TRAIN:
				if (SimulationStep >= maxStep)
				{
					double fitval = tempFitness / SimulationStep;
					//double fitval = tempFitness;
					ofstream outputFile;
					outputFile.open("fit.txt", ios_base::app);
					outputFile << fitval << endl;
					outputFile.close();
					exit(0);
				}
#endif
				
				// make oneStep false, so that the simulation is paused
				// and waiting next press of the button:
				if (oneStep)
				{
					oneStep = false;
					pause = true;
				}
			}// END NO VIDEO LOOP
		}// END if(!pause && oneStep)

		//optional but useful: debug drawing
		m_dynamicsWorld->debugDrawWorld();
#ifdef JOINT
		//saving the joint angle values to a file:
		if (SimulationStep >= maxStep)
		{
			for (int i = 0; i < num_output; i++)
			{
			string fileName;
			fileName = "joint" + to_string(i + 1) + ".txt";
			m_ragdolls[0]->save_1dfileJ(joint_val[i], fileName); //-> to be used if only end of simulation fitness is reported
			}
			exit(0);
		}
#endif
	}
	renderme(); 
	glFlush();
	glutSwapBuffers();
}

void RagdollDemo::displayCallback()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

	renderme();

	//optional but useful: debug drawing
	if (m_dynamicsWorld)
		m_dynamicsWorld->debugDrawWorld();

	glFlush();
	glutSwapBuffers();
}

void RagdollDemo::keyboardCallback(unsigned char key, int x, int y)
{
	switch (key)
	{
		// ADDING A NEW RAGDOLL:
//	case 'e':
//		{
//		btVector3 startOffset(0,2,0);
//		spawnRagdoll(startOffset);
//		break;
//		}
	case 'p':
	{
		pause = !pause;
		break;
	}
	case 'o':
	{
		oneStep = true;
		break;
	}
	default:
		DemoApplication::keyboardCallback(key, x, y);
	}

	
}



void	RagdollDemo::exitPhysics()
{

	int i;

	for (i=0;i<m_ragdolls.size();i++)
	{
		RagDoll* doll = m_ragdolls[i];
		delete doll;
	}

	//cleanup in the reverse order of creation/initialization

	//remove the rigidbodies from the dynamics world and delete them
	
	for (i=m_dynamicsWorld->getNumCollisionObjects()-1; i>=0 ;i--)
	{
		btCollisionObject* obj = m_dynamicsWorld->getCollisionObjectArray()[i];
		btRigidBody* body = btRigidBody::upcast(obj);
		if (body && body->getMotionState())
		{
			delete body->getMotionState();
		}
		m_dynamicsWorld->removeCollisionObject( obj );
		delete obj;
	}

	//delete collision shapes
	for (int j=0;j<m_collisionShapes.size();j++)
	{
		btCollisionShape* shape = m_collisionShapes[j];
		delete shape;
	}

	//delete dynamics world
	delete m_dynamicsWorld;

	//delete solver
	delete m_solver;

	//delete broadphase
	delete m_broadphase;

	//delete dispatcher
	delete m_dispatcher;

	delete m_collisionConfiguration;

	
}





