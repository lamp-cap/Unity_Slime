using UnityEngine;

namespace Slime
{
    public class ControllerTest : MonoBehaviour
    {
        [Range(1, 10)]public float speed = 4;
        private Rigidbody _rb;
        // private int _controlledId = 0;
        // public GameObject prefan;
        void Start()
        {
            _rb = GetComponent<Rigidbody>();
        }

        // Update is called once per frame
        void Update()
        {
            HandleMouseInteraction();
        }
    
        private void HandleMouseInteraction()
        {
            var controlledRb = _rb;
            var velocity = speed * new Vector3(Input.GetAxis("Horizontal"), 0, Input.GetAxis("Vertical")).normalized;
            velocity.y = controlledRb.velocity.y;
        
            if (Input.GetKeyDown(KeyCode.Space))
                velocity += new Vector3(0, 4, 0);
        
            controlledRb.velocity = velocity;
        }
    }
}
