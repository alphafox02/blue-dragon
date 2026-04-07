// Generated WHAD protocol modules (prost-build output)

pub mod ble;
pub mod discovery;
pub mod generic;

// Root message (from _.rs, inlined to avoid super:: resolution issues)
pub mod root {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Message {
        #[prost(oneof = "message::Msg", tags = "1, 2, 3")]
        pub msg: ::core::option::Option<message::Msg>,
    }
    pub mod message {
        #[derive(Clone, PartialEq, ::prost::Oneof)]
        pub enum Msg {
            #[prost(message, tag = "1")]
            Generic(crate::whad_proto::generic::Message),
            #[prost(message, tag = "2")]
            Discovery(crate::whad_proto::discovery::Message),
            #[prost(message, tag = "3")]
            Ble(crate::whad_proto::ble::Message),
        }
    }
}

pub use root::Message;
