/*
Modified from original Ragdoll Demo by Roman Popov with help from Anton Bernadsky.
June 2013 - September 2016
University of Vermont, Burlington, VT, USA.
rpopov@uvm.edu, promanev@gmail.com
*/

#include "PDSTEP_demo.h"
#include "GlutStuff.h"
#include "GLDebugDrawer.h"
#include "btBulletDynamicsCommon.h"

GLDebugDrawer	gDebugDrawer;

int main(int argc,char* argv[])
{
	int cue_time;
	RagdollDemo demoApp;
	if (argc > 1)
		cue_time = demoApp.initParams(argv[1]);

    demoApp.initPhysics();
#ifndef TRAIN
	demoApp.getDynamicsWorld()->setDebugDrawer(&gDebugDrawer);
	int retval =  glutmain(argc, argv, 640, 480, "Demo PDSTEP robot", &demoApp);
#else // TRAIN
	float ms = 1.0f / 60.0f;
	while(1)
	{
		demoApp.stepPhysics(cue_time);
	}	
	int retval = 0;
#endif //TRAIN
	return retval;
}
